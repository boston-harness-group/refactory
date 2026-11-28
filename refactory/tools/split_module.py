#!/usr/bin/env python3
"""Split a Python module into multiple modules by a mapping of names to targets.

Usage::

    python refactory/tools/split_module.py \\
        refactory/refactory_dsh/optim/base.py \\
        refactory/tools/splits/optim_base.json

The mapping file lists each top-level name and the module it moves to.
Re-running on a newer source reproduces the split; a name not in the
mapping fails loudly as unmapped, never dropped.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _import_end(tree: ast.Module) -> int:
    """1-indexed line number of the last import or module docstring."""
    last = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno or node.lineno)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            last = max(last, node.end_lineno or node.lineno)
    return last


def _docstring_end(tree: ast.Module) -> int:
    """1-indexed line number where the module docstring ends (0 if none)."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return node.end_lineno or node.lineno
    return 0


def _node_start(node: ast.AST, lines: list[str], prev_end: int) -> int:
    """1-indexed start line, including decorators and preceding ``#`` comments."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        code_line = min(d.lineno for d in node.decorator_list)
    else:
        code_line = node.lineno

    start = code_line
    for i in range(code_line - 2, prev_end - 1, -1):
        stripped = lines[i].rstrip()
        if stripped.startswith("#"):
            start = i + 1
        elif stripped == "":
            continue
        else:
            break
    return start


def _extract_nodes(
    tree: ast.Module, lines: list[str], ie: int
) -> list[dict[str, Any]]:
    """Top-level names with their 1-indexed start..end line ranges."""
    nodes: list[dict[str, Any]] = []
    prev_end = ie
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue

        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    name = t.id
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id

        if name is None:
            continue

        start = _node_start(node, lines, prev_end)
        end = node.end_lineno or node.lineno
        nodes.append({"name": name, "start": start, "end": end})
        prev_end = end

    return nodes


# ---------------------------------------------------------------------------
# Import adjustment
# ---------------------------------------------------------------------------

def _adjust_relative_imports(text: str, extra_depth: int) -> str:
    """Add *extra_depth* dots to every relative ``from .xxx import`` line."""
    if extra_depth <= 0:
        return text
    return re.sub(
        r"^(\s*from\s+)(\.+)",
        lambda m: m.group(1) + "." * (len(m.group(2)) + extra_depth),
        text,
        flags=re.MULTILINE,
    )


def _module_depth(source_path: str, target_path: str) -> int:
    """Extra directory depth of *target_path* relative to *source_path*."""
    sp = Path(source_path).resolve().parent.parts
    tp = Path(target_path).resolve().parent.parts
    return max(0, len(tp) - len(sp))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _node_text(node: dict[str, Any], lines: list[str]) -> str:
    """The verbatim source text of *node*'s line range."""
    return "".join(lines[node["start"] - 1 : node["end"]])


def _write_target(
    target_path: Path,
    docstring: str,
    import_lines: str,
    nodes: list[dict[str, Any]],
    source_lines: list[str],
    extra_depth: int,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [f'"""{docstring}"""\n\n']
    parts.append(_adjust_relative_imports(import_lines, extra_depth))

    for node in nodes:
        txt = _node_text(node, source_lines)
        parts.append("\n\n")
        parts.append(txt)
        if not txt.endswith("\n"):
            parts.append("\n")

    target_path.write_text("".join(parts))


def _write_source(
    source_path: Path,
    docstring_text: str,
    import_lines: str,
    source_key: str,
    groups: dict[str, list[dict[str, Any]]],
    modules: dict[str, str],
    source_lines: list[str],
) -> None:
    """Rewrite the source file: original docstring + imports + re-exports + remaining nodes."""
    parts: list[str] = [docstring_text, "\n", import_lines]

    for target_key in sorted(groups):
        if target_key == source_key:
            continue
        target_names = sorted(n["name"] for n in groups[target_key])
        target_abs = Path(modules[target_key]).resolve()
        target_rel = target_abs.relative_to(source_path.parent)
        dotted = "." + ".".join(target_rel.with_suffix("").parts)
        parts.append(f"\nfrom {dotted} import (  # noqa: F401, E402\n")
        for name in target_names:
            parts.append(f"    {name},\n")
        parts.append(")\n")

    for node in groups.get(source_key, []):
        parts.append("\n\n")
        txt = _node_text(node, source_lines)
        parts.append(txt)
        if not txt.endswith("\n"):
            parts.append("\n")

    source_path.write_text("".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source.py> <mapping.json>", file=sys.stderr)
        sys.exit(1)

    source_path = Path(sys.argv[1]).resolve()
    mapping_path = Path(sys.argv[2]).resolve()

    source = source_path.read_text()
    source_lines = source.splitlines(keepends=True)
    mapping = json.loads(mapping_path.read_text())
    tree = ast.parse(source, filename=str(source_path))

    # -- preamble --
    ie = _import_end(tree)
    de = _docstring_end(tree)
    docstring_text = "".join(source_lines[:de])
    import_lines = "".join(source_lines[de:ie])

    # -- nodes --
    nodes = _extract_nodes(tree, source_lines, ie)

    # -- validate --
    name_to_target: dict[str, str] = mapping["names"]
    source_names = {n["name"] for n in nodes}
    mapped_names = set(name_to_target)
    unmapped = source_names - mapped_names
    if unmapped:
        print(
            f"FATAL: unmapped names (add them to the mapping): {sorted(unmapped)}",
            file=sys.stderr,
        )
        sys.exit(1)
    extra = mapped_names - source_names
    if extra:
        print(
            f"WARNING: names in mapping not found in source: {sorted(extra)}",
            file=sys.stderr,
        )

    # -- group by target --
    groups: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        target = name_to_target[node["name"]]
        groups.setdefault(target, []).append(node)

    # -- identify source key --
    modules: dict[str, str] = mapping["modules"]
    source_key = ""
    for key, path in modules.items():
        if Path(path).resolve() == source_path:
            source_key = key
            break
    if not source_key:
        for key, path in modules.items():
            if source_path.name == Path(path).name and "base" in key:
                source_key = key
                break
    if not source_key:
        print("FATAL: cannot identify source module in the mapping", file=sys.stderr)
        sys.exit(1)

    docstrings: dict[str, str] = mapping.get("docstrings", {})

    # -- write targets --
    for target_key in sorted(groups):
        if target_key == source_key:
            continue
        target_path = Path(modules[target_key]).resolve()
        depth = _module_depth(str(source_path), str(target_path))
        ds = docstrings.get(target_key, f"Split from {source_path.name}.")
        _write_target(
            target_path, ds, import_lines,
            groups[target_key], source_lines, depth,
        )
        print(f"  wrote {target_path.relative_to(Path.cwd())} ({len(groups[target_key])} names)")

    # -- write source --
    _write_source(
        source_path, docstring_text, import_lines, source_key, groups,
        modules, source_lines,
    )
    remaining = len(groups.get(source_key, []))
    moved = sum(len(v) for k, v in groups.items() if k != source_key)
    print(f"  wrote {source_path.relative_to(Path.cwd())} (re-exports for {moved} names, {remaining} kept)")

    # -- create __init__.py for new packages --
    seen_inits: set[Path] = set()
    for target_key, path_str in modules.items():
        init = Path(path_str).resolve().parent / "__init__.py"
        if init.parent == source_path.parent:
            continue
        if init in seen_inits or init.exists():
            continue
        seen_inits.add(init)
        init.parent.mkdir(parents=True, exist_ok=True)
        pkg_name = init.parent.name
        init.write_text(f'"""``{pkg_name}`` — split from {source_path.name}."""\n')
        print(f"  wrote {init.relative_to(Path.cwd())}")

    total = sum(len(v) for v in groups.values())
    print(f"\nDone: {total} names across {len(groups)} modules.")


if __name__ == "__main__":
    main()
