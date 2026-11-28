"""Split a single-file module into a package of single-purpose modules.

Usage::

    python -m refactory_dsh.tools.split_module \\
        refactory/refactory_dsh/loop.py \\
        refactory/refactory_dsh/tools/splits/loop.json \\
        refactory/refactory_dsh/loop/

The mapping file says where each top-level name goes (for a class whose methods
are split into mixin classes, it also says where each method goes).  The script
copies source text *verbatim* by AST-node line ranges: decorators and the
comment lines directly above a node are included.  It writes each module's
imports (the original import block, to be cleaned by ruff), writes the mixin
class headers, writes the ``__init__.py`` with the Loop class inheriting all
mixins, and writes the re-exports.

A name or method present in the source but absent from the mapping fails
loudly.  Re-running on a newer source with the same mapping reproduces the
split; a name added since fails as unmapped.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _leading_comments(source_lines: list[str], node_start: int) -> int:
    """Return the first line (1-based) of comments directly above *node_start*."""
    idx = node_start - 2  # 0-based index of line before node
    while idx >= 0 and (source_lines[idx].strip().startswith("#") or source_lines[idx].strip() == ""):
        if source_lines[idx].strip() == "":
            # An empty line breaks the comment block only if the line above it
            # is NOT a comment (i.e. two blank lines or blank then code).
            if idx - 1 >= 0 and source_lines[idx - 1].strip().startswith("#"):
                idx -= 1
                continue
            break
        idx -= 1
    return idx + 2  # back to 1-based


def _decorator_start(node: ast.AST) -> int:
    """First line of decorators, or the node's own line."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _node_lines(source_lines: list[str], node: ast.AST) -> tuple[int, int]:
    """(first_line, last_line) 1-based, including leading comments and decorators."""
    first = min(_leading_comments(source_lines, _decorator_start(node)), _decorator_start(node))
    last = node.end_lineno
    return first, last


def _extract(source_lines: list[str], first: int, last: int) -> str:
    """Extract lines [first, last] (1-based inclusive) from source."""
    return "\n".join(source_lines[first - 1 : last])


# ---------------------------------------------------------------------------
# Import block extraction
# ---------------------------------------------------------------------------

def _import_block(tree: ast.Module, source_lines: list[str]) -> str:
    """The file's import block: everything from line 1 up to and including the
    last import/importfrom/if-TYPE_CHECKING, plus the module docstring."""
    last_import = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = max(last_import, node.end_lineno)
        elif isinstance(node, ast.If):
            # TYPE_CHECKING block
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
               (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
                last_import = max(last_import, node.end_lineno)
    if last_import == 0:
        return ""
    return "\n".join(source_lines[0:last_import])


# ---------------------------------------------------------------------------
# Main splitting logic
# ---------------------------------------------------------------------------

MIXIN_CLASS_NAMES = {
    "resume": "_ResumeMixin",
    "parts": "_PartsMixin",
    "forward": "_ForwardMixin",
    "evaluate": "_EvaluateMixin",
    "incidents": "_IncidentsMixin",
    "selection": "_SelectionMixin",
    "context": "_ContextMixin",
    "report": "_ReportMixin",
}


PATCH_HAZARDS: dict[str, str] = {
    # name -> the dotted module path tests patch it through
    # Methods in other modules must access these through the package module
    # object (sys.modules) so monkeypatch on the old path still lands.
    "_new_steps": "refactory_dsh.loop",
}


def _collect_top_level(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """(name, node) for every top-level definition, in source order."""
    items = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            items.append((node.name, node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    items.append((t.id, node))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            items.append((node.target.id, node))
        elif isinstance(node, ast.If):
            # Only the ``if __name__ == "__main__"`` block; TYPE_CHECKING
            # is part of the import header and handled by the import-block
            # extraction.
            test = node.test
            is_type_checking = (
                (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
            )
            if not is_type_checking:
                items.append(("__if_block__", node))
    return items


def _collect_class_members(
    cls_node: ast.ClassDef,
) -> list[tuple[str, ast.AST]]:
    """(key, node) for each member of a class, in source order.

    Methods get their name as key; class variables get ``VAR:<name>``."""
    members = []
    for node in ast.iter_child_nodes(cls_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.append((node.name, node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    members.append((f"VAR:{t.id}", node))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.append((f"VAR:{node.target.id}", node))
    return members


def _local_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names that are local to a function: parameters, nested defs, assignments."""
    names: set[str] = set()
    # Parameters
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        names.add(arg.arg)
    if func_node.args.vararg:
        names.add(func_node.args.vararg.arg)
    if func_node.args.kwarg:
        names.add(func_node.args.kwarg.arg)
    # Assignments and comprehension variables in the body
    for child in ast.walk(func_node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child is not func_node:
                names.add(child.name)
        elif isinstance(child, ast.For):
            if isinstance(child.target, ast.Name):
                names.add(child.target.id)
    return names


def _cross_module_globals(
    cls_node: ast.ClassDef,
    top_level_targets: dict[str, str],
    method_targets: dict[str, str],
) -> dict[str, set[str]]:
    """For each method, the top-level names it references that live in a different module.

    Returns {method_name: {global_name, ...}}.  Excludes names that are
    parameters or local variables of the method."""
    result: dict[str, set[str]] = {}
    for node in ast.iter_child_nodes(cls_node):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        method_target = method_targets.get(node.name, "__init__")
        locals_ = _local_names(node)
        refs = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in top_level_targets:
                if child.id in locals_:
                    continue
                ref_target = top_level_targets[child.id]
                if ref_target != method_target and method_target != "__init__":
                    refs.add(child.id)
        if refs:
            result[node.name] = refs
    return result


def split_module(
    source_path: Path,
    mapping_path: Path,
    output_dir: Path,
) -> None:
    source_text = source_path.read_text()
    source_lines = source_text.splitlines()
    tree = ast.parse(source_text)
    mapping = json.loads(mapping_path.read_text())

    top_level_map: dict[str, str] = mapping["top_level"]
    method_map: dict[str, str] = mapping["loop_methods"]

    # ── validate: every name in source is mapped ──────────────────────
    top_items = _collect_top_level(tree)
    unmapped_top = []
    for name, _node in top_items:
        if name == "__if_block__":
            continue
        if name not in top_level_map:
            unmapped_top.append(name)
    if unmapped_top:
        raise SystemExit(f"unmapped top-level names: {', '.join(unmapped_top)}")

    loop_node = None
    for name, node in top_items:
        if name == "Loop" and isinstance(node, ast.ClassDef):
            loop_node = node
            break
    if loop_node is None:
        raise SystemExit("Loop class not found in source")

    class_members = _collect_class_members(loop_node)
    unmapped_methods = []
    for key, _node in class_members:
        if key not in method_map:
            unmapped_methods.append(key)
    if unmapped_methods:
        raise SystemExit(f"unmapped Loop members: {', '.join(unmapped_methods)}")

    # ── validate: every mapped name exists in source ──────────────────
    source_top_names = {name for name, _ in top_items if name != "__if_block__"}
    for name in top_level_map:
        if name not in source_top_names:
            raise SystemExit(f"mapped top-level name not in source: {name}")
    source_method_keys = {key for key, _ in class_members}
    for key in method_map:
        if key not in source_method_keys:
            raise SystemExit(f"mapped Loop member not in source: {key}")

    # ── group items by target module ──────────────────────────────────
    # Top-level items
    module_top_items: dict[str, list[tuple[str, ast.AST]]] = {}
    for name, node in top_items:
        if name == "__if_block__":
            continue
        target = top_level_map[name]
        module_top_items.setdefault(target, []).append((name, node))

    # Loop methods by mixin module
    module_methods: dict[str, list[tuple[str, ast.AST]]] = {}
    for key, node in class_members:
        target = method_map[key]
        module_methods.setdefault(target, []).append((key, node))

    # ── compute which globals each mixin module needs to import ───────
    cross_refs = _cross_module_globals(loop_node, top_level_map, method_map)

    # For each mixin module, collect the config/other-module names its methods need
    module_needed_imports: dict[str, set[str]] = {}
    for key, _node in class_members:
        if not key.startswith("VAR:"):
            method_name = key
            target = method_map[key]
            if target == "__init__":
                continue
            if method_name in cross_refs:
                module_needed_imports.setdefault(target, set()).update(cross_refs[method_name])

    # Also check top-level functions/classes for cross-module name references
    # (e.g. type annotations using names defined in other modules)
    for name, node in top_items:
        if name == "__if_block__" or name == "Loop":
            continue
        target = top_level_map.get(name)
        if target is None or target == "__init__":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in top_level_map:
                ref_target = top_level_map[child.id]
                if ref_target != target:
                    module_needed_imports.setdefault(target, set()).add(child.id)

    # ── get the module docstring ──────────────────────────────────────
    docstring = ast.get_docstring(tree, clean=False)
    docstring_end = 0
    if docstring:
        first_node = tree.body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
            docstring_end = first_node.end_lineno

    # ── get the import block (without the docstring) ────────────────
    last_import_line = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_line = max(last_import_line, node.end_lineno)
        elif isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
               (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
                last_import_line = max(last_import_line, node.end_lineno)
    # Start after the docstring (if any), include up to last import
    import_start = docstring_end  # 0-based: line after docstring
    imports = "\n".join(source_lines[import_start:last_import_line])

    # ── identify all target modules ───────────────────────────────────
    all_modules = set()
    all_modules.update(top_level_map.values())
    all_modules.update(method_map.values())
    all_modules.discard("__init__")

    # ── write each module file ────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which names from config (and other modules) each module needs
    # so we can add specific imports
    for mod_name in sorted(all_modules):
        lines: list[str] = []
        # The import block (ruff will clean unused)
        lines.append(imports)
        lines.append("")

        # Additional imports from sibling modules for cross-references
        needed = module_needed_imports.get(mod_name, set())
        # Group by source module
        imports_by_source: dict[str, list[str]] = {}
        for ref_name in sorted(needed):
            ref_module = top_level_map[ref_name]
            if ref_name in PATCH_HAZARDS:
                # Patch-hazardous names: accessed through sys.modules at call time
                # (handled inline in the method bodies)
                continue
            imports_by_source.setdefault(ref_module, []).append(ref_name)

        for src_mod, names in sorted(imports_by_source.items()):
            if src_mod == mod_name:
                continue
            names_str = ", ".join(sorted(names))
            lines.append(f"from refactory_dsh.loop.{src_mod} import {names_str}")

        if imports_by_source:
            lines.append("")

        # Top-level items for this module
        top_for_mod = module_top_items.get(mod_name, [])
        for name, node in top_for_mod:
            first, last = _node_lines(source_lines, node)
            lines.append("")
            lines.append(_extract(source_lines, first, last))

        # Mixin class if this module has Loop methods
        methods_for_mod = module_methods.get(mod_name, [])
        if methods_for_mod:
            mixin_name = MIXIN_CLASS_NAMES.get(mod_name)
            if mixin_name is None:
                raise SystemExit(f"no mixin class name defined for module: {mod_name}")
            lines.append("")
            lines.append("")
            lines.append(f"class {mixin_name}:")

            # Determine if this mixin has any patch-hazardous globals
            hazardous_in_mod = set()
            for key, node in methods_for_mod:
                if not key.startswith("VAR:"):
                    if key in cross_refs:
                        for ref in cross_refs[key]:
                            if ref in PATCH_HAZARDS:
                                hazardous_in_mod.add(ref)

            for key, node in methods_for_mod:
                if key.startswith("VAR:"):
                    # Class variable
                    first, last = _node_lines(source_lines, node)
                    text = _extract(source_lines, first, last)
                    # Indent class-level (they were already indented in the
                    # original class; extract them without the class-level indent
                    # and re-indent with 4 spaces)
                    # The original source has 4-space indent for class body
                    for tl in text.splitlines():
                        lines.append(f"    {tl.strip()}" if tl.strip() else "")
                else:
                    # Method
                    first, last = _node_lines(source_lines, node)
                    text = _extract(source_lines, first, last)
                    lines.append("")
                    # Methods are already indented with 4 spaces in the original
                    for tl in text.splitlines():
                        lines.append(tl)

        lines.append("")
        content = "\n".join(lines)

        # Handle patch-hazardous globals: replace direct name references with
        # sys.modules lookup.  Only in method bodies of this module's mixin.
        if methods_for_mod:
            hazardous_used = set()
            for key, _node in methods_for_mod:
                if not key.startswith("VAR:") and key in cross_refs:
                    for ref in cross_refs[key]:
                        if ref in PATCH_HAZARDS:
                            hazardous_used.add(ref)
            for hname in hazardous_used:
                pkg = PATCH_HAZARDS[hname]
                # Replace bare calls to the hazardous name with a sys.modules
                # lookup.  We only replace at call sites (name followed by '(')
                # to avoid false positives.
                content = content.replace(
                    f"{hname}(",
                    f'sys.modules["{pkg}"].{hname}(',
                )

        # Fix relative imports inside method bodies: when the original file
        # was at refactory_dsh/loop.py, ``from .foo`` resolved to
        # ``refactory_dsh.foo``.  Now that the file is inside
        # refactory_dsh/loop/, ``from .foo`` would resolve to
        # ``refactory_dsh.loop.foo`` which is wrong.  Convert them to
        # absolute imports.  Only affects lazy imports in method bodies
        # (indented); the top-level import block is already absolute.
        parent_package = source_path.parent.name  # "refactory_dsh"
        # Match indented relative imports that are NOT to sibling modules
        sibling_names = "|".join(sorted(all_modules))
        content = re.sub(
            r"^(\s+from) \.((?!" + sibling_names + r")\w+)",
            rf"\1 {parent_package}.\2",
            content,
            flags=re.MULTILINE,
        )

        (output_dir / f"{mod_name}.py").write_text(content)

    # ── write __init__.py ─────────────────────────────────────────────
    init_lines: list[str] = []
    # Module docstring
    if docstring:
        init_lines.append(f'"""{docstring}"""')
        init_lines.append("")
    init_lines.append(imports)
    init_lines.append("")

    # Import from each sub-module so old paths keep working
    # Collect all names by their source module for re-export
    reexport_by_module: dict[str, list[str]] = {}
    for name, target in sorted(top_level_map.items()):
        if target == "__init__":
            continue
        reexport_by_module.setdefault(target, []).append(name)

    # Also re-export mixin classes
    for mod_name in sorted(all_modules):
        mixin_name = MIXIN_CLASS_NAMES.get(mod_name)
        if mixin_name:
            reexport_by_module.setdefault(mod_name, []).append(mixin_name)

    for mod_name, names in sorted(reexport_by_module.items()):
        names_str = ", ".join(sorted(names))
        init_lines.append(f"from refactory_dsh.loop.{mod_name} import {names_str}")
    init_lines.append("")

    # Top-level items that belong in __init__
    init_top = module_top_items.get("__init__", [])
    for name, node in init_top:
        if name == "Loop":
            continue  # written separately below
        first, last = _node_lines(source_lines, node)
        init_lines.append("")
        init_lines.append(_extract(source_lines, first, last))

    # The Loop class: docstring, class vars, and __init__-targeted methods,
    # inheriting all mixins.
    init_methods = module_methods.get("__init__", [])
    mixin_bases = []
    for mod_name in ("resume", "parts", "forward", "evaluate", "incidents",
                     "selection", "context", "report"):
        if mod_name in module_methods:
            mixin_bases.append(MIXIN_CLASS_NAMES[mod_name])

    # Get the Loop class's original bases
    loop_bases_src = []
    for base in loop_node.bases:
        loop_bases_src.append(ast.unparse(base))

    all_bases = mixin_bases + loop_bases_src
    bases_str = ", ".join(all_bases)

    init_lines.append("")
    init_lines.append("")

    # Get Loop's docstring
    loop_docstring = ast.get_docstring(loop_node, clean=False)
    init_lines.append(f"class Loop({bases_str}):")
    if loop_docstring:
        init_lines.append(f'    """{loop_docstring}"""')
        init_lines.append("")

    for key, node in init_methods:
        if key.startswith("VAR:"):
            first, last = _node_lines(source_lines, node)
            text = _extract(source_lines, first, last)
            for tl in text.splitlines():
                init_lines.append(f"    {tl.strip()}" if tl.strip() else "")
        else:
            first, last = _node_lines(source_lines, node)
            text = _extract(source_lines, first, last)
            init_lines.append("")
            for tl in text.splitlines():
                init_lines.append(tl)

    # Write the if __name__ == "__main__" block
    for name, node in top_items:
        if name == "__if_block__":
            first, last = _node_lines(source_lines, node)
            init_lines.append("")
            init_lines.append("")
            init_lines.append(_extract(source_lines, first, last))

    init_lines.append("")
    (output_dir / "__init__.py").write_text("\n".join(init_lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="path to the source .py file")
    parser.add_argument("mapping", help="path to the JSON mapping file")
    parser.add_argument("output", help="path to the output package directory")
    args = parser.parse_args(argv)
    split_module(Path(args.source), Path(args.mapping), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
