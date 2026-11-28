#!/usr/bin/env python3
"""Split a Python module into multiple modules by a mapping of names to targets.

Usage::

    python refactory/tools/split_module.py \\
        refactory/refactory_dsh/optim/base.py \\
        refactory/tools/splits/optim_base.json

The mapping file lists each top-level name and the module it moves to.
Re-running on a newer source reproduces the split; a name not in the
mapping fails loudly as unmapped, never dropped.

The mapping may also include ``cross_imports`` — imports between the new
modules for names that were same-module globals before the split.  Three
kinds are supported:

- **top_imports**: placed right after the shared import block (safe sources
  that have no outgoing deps, like ``types`` or ``context``).
- **bottom_imports**: placed at EOF with ``# noqa: E402`` (between sibling
  modules that form cycles if imported at the top level).
- **lazy_imports**: placed inside function/method bodies, right after the
  docstring (required when the target imports from this module at the top
  level, so a module-level import would create a cycle).
- **deferred_names**: top-level assignments that depend on bottom-imported
  names and must be placed after the bottom imports.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LINE_LENGTH = 88


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
        nodes.append({"name": name, "start": start, "end": end, "_ast": node})
        prev_end = end

    return nodes


# ---------------------------------------------------------------------------
# Import analysis
# ---------------------------------------------------------------------------

def _import_statements(tree: ast.Module, lines: list[str]) -> list[dict[str, Any]]:
    """Parse the module-level import block into individual import entries."""
    stmts: list[dict[str, Any]] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Import):
            names = {a.asname or a.name.split(".")[-1] for a in node.names}
            module = node.names[0].name
            is_from = False
        else:
            if node.names[0].name == "*":
                names = {"*"}
            else:
                names = {a.asname or a.name for a in node.names}
            module = "." * node.level + (node.module or "")
            is_from = True
        text = "".join(lines[node.lineno - 1 : node.end_lineno])
        stmts.append({
            "names": names,
            "text": text,
            "module": module,
            "is_from": is_from,
            "lineno": node.lineno,
            "end_lineno": node.end_lineno,
        })
    return stmts


def _names_used_by(nodes: list[dict[str, Any]], source: str) -> set[str]:
    """Names that need a module-level import for a group of top-level definitions.

    Excludes names that are locally imported inside every node that uses them
    (those are handled by in-body imports, not the shared import block).
    """
    tree = ast.parse(source)
    name_set = {n["name"] for n in nodes}
    used: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        node_name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_name = node.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    node_name = t.id
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            node_name = node.target.id
        if node_name not in name_set:
            continue

        local_imports: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for a in child.names:
                    local_imports.add(a.asname or a.name.split(".")[-1])
            elif isinstance(child, ast.ImportFrom):
                for a in child.names:
                    if a.name != "*":
                        local_imports.add(a.asname or a.name)

        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id not in local_imports:
                used.add(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    if child.value.id not in local_imports:
                        used.add(child.value.id)

    return used


_STDLIB_TOP = frozenset({
    "abc", "ast", "asyncio", "base64", "bisect", "builtins", "calendar",
    "cmath", "codecs", "collections", "colorsys", "concurrent", "configparser",
    "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime",
    "decimal", "difflib", "dis", "email", "encodings", "enum", "errno",
    "faulthandler", "filecmp", "fileinput", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob",
    "graphlib", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "imaplib", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "keyword", "linecache", "locale", "logging", "lzma",
    "mailbox", "marshal", "math", "mimetypes", "multiprocessing",
    "numbers", "operator", "os", "pathlib", "pdb", "pickle", "pkgutil",
    "platform", "plistlib", "poplib", "posixpath", "pprint",
    "profile", "pstats", "pty", "pwd", "py_compile", "pydoc",
    "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "sqlite3",
    "ssl", "stat", "statistics", "string", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "tomllib", "trace", "traceback", "tracemalloc", "tty", "turtle",
    "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
    "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
})


def _import_group_key(stmt: dict[str, Any]) -> int:
    mod = stmt["module"]
    text = stmt["text"].lstrip()
    if text.startswith("from __future__"):
        return 0
    top = mod.split(".")[0]
    if mod.startswith("."):
        return 3
    if top in _STDLIB_TOP:
        return 1
    return 2


def _import_sort_key(stmt: dict[str, Any]) -> tuple:
    """Sort key within a group: bare imports first, then from-imports, alphabetical."""
    return (stmt["is_from"], stmt["module"].lower())


def _trim_import(text: str, keep: set[str]) -> str:
    """Remove unused names from a multi-name import statement."""
    m = re.match(r"(from\s+\S+\s+import\s+)\(([^)]+)\)", text, re.DOTALL)
    if m:
        prefix = m.group(1)
        names_text = m.group(2)
        kept = []
        for part in names_text.split(","):
            name = part.strip().rstrip(",")
            if not name:
                continue
            as_m = re.match(r"(\w+)\s+as\s+(\w+)", name)
            check_name = as_m.group(2) if as_m else name
            if check_name in keep:
                kept.append(name)
        if len(kept) == 1:
            return f"{prefix}{kept[0]}\n"
        return prefix + "(\n" + "".join(f"    {n},\n" for n in kept) + ")\n"

    m = re.match(r"((?:from\s+\S+\s+)?import\s+)(.+)", text.rstrip())
    if m:
        prefix = m.group(1)
        names_part = m.group(2)
        kept = []
        for part in names_part.split(","):
            name = part.strip()
            if not name:
                continue
            as_m = re.match(r"(\w+)\s+as\s+(\w+)", name)
            check_name = as_m.group(2) if as_m else name.split(".")[-1]
            if check_name in keep:
                kept.append(name)
        return prefix + ", ".join(kept) + "\n"

    return text


def _build_import_block(
    stmts: list[dict[str, Any]],
    used_names: set[str],
    extra_depth: int,
) -> str:
    """Build a clean, sorted import block from the shared imports.

    Keeps only imports that provide at least one used name, trims multi-name
    imports, applies depth adjustment, and sorts within each group (future,
    stdlib, third-party, local).  Returns the text ready to write.
    """
    kept: list[dict[str, Any]] = []
    for stmt in stmts:
        provided = stmt["names"]
        is_future = "from __future__" in stmt["text"]
        needed = provided & used_names
        if not needed and not is_future and "*" not in provided:
            continue

        text = stmt["text"]
        if needed != provided and len(provided) > 1 and not is_future:
            text = _trim_import(text, needed)

        text = _adjust_relative_imports(text, extra_depth)

        kept.append({**stmt, "text": text})

    kept.sort(key=lambda s: (_import_group_key(s), _import_sort_key(s)))

    parts: list[str] = []
    prev_group: int | None = None
    for stmt in kept:
        group = _import_group_key(stmt)
        if prev_group is not None and group != prev_group:
            parts.append("\n")
        prev_group = group
        text = stmt["text"]
        parts.append(text)
        if not text.endswith("\n"):
            parts.append("\n")

    return "".join(parts)


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


def _wrap_long_imports(text: str) -> str:
    """Wrap in-body ``from X import ...`` lines that exceed LINE_LENGTH."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip()
        if (len(stripped) > LINE_LENGTH
                and re.match(r"\s+from\s+\S+\s+import\s+\w+.*,", stripped)
                and "(" not in stripped):
            m = re.match(r"(\s*)(from\s+\S+\s+import\s+)(.*)", stripped)
            if m:
                indent = m.group(1)
                prefix = m.group(2)
                names = [n.strip() for n in m.group(3).split(",") if n.strip()]
                out.append(f"{indent}{prefix}(\n")
                for n in names:
                    out.append(f"{indent}    {n},\n")
                out.append(f"{indent})\n")
                continue
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# Cross-module import rendering
# ---------------------------------------------------------------------------

def _render_import_line(
    module: str, names: list[str], noqa: str = "",
) -> str:
    """Render an import statement, using multi-line form when it exceeds LINE_LENGTH."""
    suffix = f"  {noqa}" if noqa else ""
    single = f"from {module} import {', '.join(names)}{suffix}\n"
    if len(single.rstrip()) <= LINE_LENGTH:
        return single
    head = f"from {module} import ({suffix}\n"
    body = "".join(f"    {n},\n" for n in names)
    return head + body + ")\n"


def _render_top_imports(top_imports: dict[str, list[str]]) -> str:
    """Render top-level cross-module imports.

    Preserves the order given in the mapping (not sorted), so the output is
    stable regardless of ruff isort's grouping rules.
    """
    parts: list[str] = []
    for module in top_imports:
        names = top_imports[module]
        parts.append(_render_import_line(module, names))
    return "".join(parts)


def _render_bottom_imports(bottom_imports: dict[str, list[str]]) -> str:
    """Render bottom-of-file cross-module imports with # noqa: E402."""
    parts = ["\n\n# --- cross-module imports (resolved at definition time, used at call time) ---\n"]
    for module in sorted(bottom_imports):
        names = sorted(bottom_imports[module])
        parts.append(_render_import_line(module, names, noqa="# noqa: E402"))
    return "".join(parts)


def _inject_lazy_imports(
    text: str, lazy_imports: dict[str, dict[str, list[str]]],
) -> str:
    """Insert lazy imports into function bodies, right after each function's docstring."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []

    func_nodes: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in lazy_imports:
                func_nodes[node.name] = node

    for func_name, imports_by_module in lazy_imports.items():
        node = func_nodes.get(func_name)
        if node is None:
            print(
                f"WARNING: lazy_imports target function {func_name!r} not found",
                file=sys.stderr,
            )
            continue

        body_indent = " " * (node.col_offset + 4)
        ds = ast.get_docstring(node, clean=False)
        if ds:
            insert_line = node.body[0].end_lineno
        else:
            insert_line = node.lineno

        import_lines: list[str] = []
        for module in sorted(imports_by_module):
            names = sorted(imports_by_module[module])
            import_lines.append(
                f"{body_indent}from {module} import {', '.join(names)}\n"
            )

        insertions.append((insert_line, "".join(import_lines)))

    if not insertions:
        return text

    insertions.sort(key=lambda x: x[0], reverse=True)
    for after_line, block in insertions:
        lines.insert(after_line, block)

    return "".join(lines)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _node_text(node: dict[str, Any], lines: list[str]) -> str:
    """The verbatim source text of *node*'s line range."""
    return "".join(lines[node["start"] - 1 : node["end"]])


def _write_target(
    target_path: Path,
    docstring: str,
    import_block: str,
    nodes: list[dict[str, Any]],
    source_lines: list[str],
    extra_depth: int,
    cross_imports: dict[str, Any] | None = None,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [f'"""{docstring}"""\n\n\n']
    parts.append(import_block)

    if cross_imports and "top_imports" in cross_imports:
        parts.append("\n")
        parts.append(_render_top_imports(cross_imports["top_imports"]))

    deferred = set(cross_imports.get("deferred_names", [])) if cross_imports else set()

    first_code = True
    for node in nodes:
        if node["name"] in deferred:
            parts.append("\n\n")
            parts.append(
                f"# {node['name']} is defined after cross-module imports, below.\n"
            )
            continue
        txt = _node_text(node, source_lines)
        txt = _adjust_relative_imports(txt, extra_depth)
        if first_code:
            is_def = isinstance(
                node["_ast"], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
            parts.append("\n\n" if is_def else "\n")
            first_code = False
        else:
            parts.append("\n\n")
        parts.append(txt)
        if not txt.endswith("\n"):
            parts.append("\n")

    text = "".join(parts)

    if extra_depth > 0:
        text = _wrap_long_imports(text)

    if cross_imports and "lazy_imports" in cross_imports:
        text = _inject_lazy_imports(text, cross_imports["lazy_imports"])

    if cross_imports and "bottom_imports" in cross_imports:
        text += _render_bottom_imports(cross_imports["bottom_imports"])

    if deferred:
        for node in nodes:
            if node["name"] not in deferred:
                continue
            txt = _node_text(node, source_lines)
            txt = _adjust_relative_imports(txt, extra_depth)
            txt_lines = txt.split("\n")
            out_lines: list[str] = []
            for tl in txt_lines:
                if re.match(r"\w+\s*=", tl) and "# noqa:" not in tl:
                    out_lines.append(tl.rstrip() + "  # noqa: E402")
                else:
                    out_lines.append(tl)
            text += "\n" + "\n".join(out_lines)

    target_path.write_text(text)


def _reexport_sort_key(name: str) -> tuple:
    """Sort re-exports: _CONST, CONST, _Class, Class, _func, func — case-insensitive."""
    stripped = name.lstrip("_")
    has_us = name.startswith("_")
    if stripped and stripped[0].isupper():
        is_allcaps = all(c.isupper() or c in "_0123456789" for c in stripped)
        if is_allcaps:
            group = 0 if has_us else 1
        else:
            group = 2 if has_us else 3
    else:
        group = 4 if has_us else 5
    return (group, name.lower())


def _write_source(
    source_path: Path,
    docstring_text: str,
    import_stmts: list[dict[str, Any]],
    source_key: str,
    groups: dict[str, list[dict[str, Any]]],
    modules: dict[str, str],
    source_lines: list[str],
    source: str,
) -> None:
    """Rewrite the source file: filtered imports + re-exports + remaining nodes."""
    parts: list[str] = [docstring_text, "\n\n"]

    remaining_nodes = groups.get(source_key, [])
    used = _names_used_by(remaining_nodes, source)
    import_block = _build_import_block(import_stmts, used, 0)
    parts.append(import_block)

    for target_key in sorted(groups):
        if target_key == source_key:
            continue
        target_names = sorted(
            (n["name"] for n in groups[target_key]),
            key=_reexport_sort_key,
        )
        target_abs = Path(modules[target_key]).resolve()
        target_rel = target_abs.relative_to(source_path.parent)
        dotted = "." + ".".join(target_rel.with_suffix("").parts)
        parts.append(f"from {dotted} import (  # noqa: F401, E402\n")
        for name in target_names:
            parts.append(f"    {name},\n")
        parts.append(")\n")

    for node in remaining_nodes:
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

    # -- import statements --
    import_stmts = _import_statements(tree, source_lines)

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
    cross_imports_all: dict[str, Any] = mapping.get("cross_imports", {})

    # -- write targets --
    for target_key in sorted(groups):
        if target_key == source_key:
            continue
        target_path = Path(modules[target_key]).resolve()
        depth = _module_depth(str(source_path), str(target_path))
        ds = docstrings.get(target_key, f"Split from {source_path.name}.")
        ci = cross_imports_all.get(target_key)

        target_nodes = groups[target_key]
        used = _names_used_by(target_nodes, source)
        import_block = _build_import_block(import_stmts, used, depth)

        _write_target(
            target_path, ds, import_block,
            target_nodes, source_lines, depth,
            cross_imports=ci,
        )
        print(f"  wrote {target_path.relative_to(Path.cwd())} ({len(target_nodes)} names)")

    # -- write source --
    _write_source(
        source_path, docstring_text, import_stmts, source_key, groups,
        modules, source_lines, source,
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

    # -- remove unused imports with ruff --
    all_files = [str(Path(p).resolve()) for p in modules.values()]
    all_files.extend(str(init) for init in seen_inits)
    result = subprocess.run(
        ["ruff", "check", "--fix", "--select", "F401", *all_files],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())

    total = sum(len(v) for v in groups.values())
    print(f"\nDone: {total} names across {len(groups)} modules.")


if __name__ == "__main__":
    main()
