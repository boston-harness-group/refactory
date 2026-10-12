"""Native Create adapters: bootstrap, study, admission, synthesis and precheck.

Generated package code is imported only by contained child processes. Admission
uses the actual native registry and validator; it never authors missing files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from refactory_dsh import registry, sandbox
from refactory_dsh.create_fidelity_prompts import (
    FACTORY_TEMPLATE_PATH, FACTORY_TEMPLATE_SOURCE_PATH,
    NATIVE_INSTRUCTIONS, factory_template, load_prompt,
)
from refactory_dsh.paths import project_for_nodes

S = ".factory/strategy/"
R = ".factory/reviews/"

NATIVE_API = """\
from refactory_dsh import nn
class RequestedMode(nn.Mode):
    name = 'run'
    description = 'Describe the actual behavior'
    unscored_reason = 'This mode produces an artifact rather than an optimization metric'
    @classmethod
    def declare(cls):
        return {'input_path': nn.Knob('', 'str', description='Runtime input file')}
    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node('compute', 'op', call='refactory_packages.PACKAGE_NAME.ops:compute',
                          knobs=('input_path',), writes=('result.json',)))
        return graph
    def invariants(self):
        return (nn.artifact_produced('result.json', by='compute'),)
PACKAGE = nn.Package(name='PACKAGE_NAME', description='Purpose', modes=(RequestedMode,),
                     ops=('refactory_packages.PACKAGE_NAME.ops',))

Operations receive RunContext. ctx.knob(name) is the invocation input;
ctx.workdir is this invocation's output directory. Return only declared facts.
Use ctx.write_json/text for declared output files. Read external runtime input
paths with pathlib.Path; do not list an external input file in node.reads, which
describes files produced inside the run. Reject invalid data with a useful
exception; native executor reports failure honestly. Keep build() pure.
Input knobs are legitimate runtime inputs even when no optimizer will run.
No score override or Task is required for a deliberately unscored portable mode.
"""


def native_instructions(ctx: Any) -> str:
    from refactory_dsh.nn.executor import render

    return render(NATIVE_INSTRUCTIONS, ctx)


def agent_prompt(ctx: Any, node: Any) -> str:
    """Supply complete declared input bodies, preserving their bytes after render.

    Report text may itself contain template-looking strings or source code. Only
    the authored instruction template is rendered; supplied input is raw data.
    """
    from refactory_dsh.nn.executor import render

    prompt = render(node.prompt_template, ctx)
    return prompt + complete_inputs(ctx, node.reads)


def complete_inputs(ctx: Any, reads: tuple[str, ...]) -> str:
    sections = []
    for relative in reads:
        path = ctx.path(relative)
        if not path.is_file():
            raise ValueError(f"candidate stage requires declared substantive input {relative}")
        sections.append(f"=== COMPLETE DECLARED INPUT {relative} ===\n" + path.read_text())
    return "\n\n" + "\n\n".join(sections) if sections else ""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_digest(root: Path) -> str:
    """Identity of authored bytes, excluding generated receipts/cache/skill."""
    hashes = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"generated package contains a symlink: {path}")
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root)
        if ".pytest_cache" in relative.parts:
            # Ignore only pytest's known cache records, not arbitrary files or
            # authored tests/data merely placed below a cache-named directory.
            cache_index = relative.parts.index(".pytest_cache")
            cache_name = "/".join(relative.parts[cache_index + 1:])
            if cache_name in {"CACHEDIR.TAG", "README.md", ".gitignore", "v/cache/nodeids", "v/cache/lastfailed", "v/cache/stepwise"}:
                continue
        if path.name in ("SKILL.md", ".native-admission.json", ".origin.json"):
            continue
        hashes[path.relative_to(root).as_posix()] = _sha(path.read_bytes())
    return _sha(json.dumps(hashes, sort_keys=True).encode())


def _name(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("package name must start with a lower-case letter and contain only letters, digits and underscores")
    return value


def _template_input() -> tuple[str, dict[str, Any]]:
    canonical = factory_template()
    text = canonical["text"]
    provenance = {"schema_version": 1, "adapter": "native-bootstrap-template",
                  "source_commit": canonical["source_commit"],
                  "source_path": canonical["source_path"],
                  "source_sha256": canonical["source_sha256"],
                  "template_path": FACTORY_TEMPLATE_PATH,
                  "template_sha256": _sha(text.encode()), "template_bytes": len(text.encode()),
                  "note": "Canonical input supplied unchanged; the later agent delivery records the filled output, not a presumed copy or acceptance result."}
    return text, provenance


def _publish_template(ctx: Any) -> None:
    text, provenance = _template_input()
    if ctx.exists(FACTORY_TEMPLATE_PATH) and ctx.path(FACTORY_TEMPLATE_PATH).read_bytes() != text.encode():
        raise ValueError("existing bootstrap template differs from the canonical source; use a fresh run directory")
    if ctx.exists(FACTORY_TEMPLATE_SOURCE_PATH) and ctx.read_json(FACTORY_TEMPLATE_SOURCE_PATH) != provenance:
        raise ValueError("existing bootstrap template provenance differs from the canonical source")
    if not ctx.exists(FACTORY_TEMPLATE_PATH):
        ctx.write_text(FACTORY_TEMPLATE_PATH, text)
    if not ctx.exists(FACTORY_TEMPLATE_SOURCE_PATH):
        ctx.write_json(FACTORY_TEMPLATE_SOURCE_PATH, provenance)


def init_user_intent(ctx: Any) -> dict[str, Any]:
    """Publish the complete request before bootstrap or any paid model call."""
    task = dict(getattr(ctx, "task", {}) or {})
    brief = str(task.get("brief") or ctx.knob("brief", "") or "").strip()
    raw_name = str(task.get("name") or ctx.knob("name", "") or "").strip()
    if not brief:
        raise ValueError("Create fidelity requires a nonempty original brief")
    if not raw_name:
        words = re.findall(r"[a-z0-9]+", brief.lower())[:3]
        raw_name = "_".join(words) or "created_pack"
        if raw_name[0].isdigit():
            raw_name = "pkg_" + raw_name
    name = _name(raw_name)
    variant = str(task.get("variant") or ctx.knob("variant", "portable") or "portable").strip()
    if variant != "portable":
        raise ValueError(f"Create fidelity variant {variant!r} is unsupported; only new portable native packages are implemented")
    if task.get("revision_of") or task.get("mode") or ctx.knob("revision_of", ""):
        raise ValueError("Create fidelity update-existing is unsupported; use public Edit")
    source = Path(str(task.get("project_path") or ctx.knob("project_path", "") or project_for_nodes())).resolve()
    if not source.is_dir():
        raise ValueError(f"source project does not exist: {source}")
    run = Path(ctx.workdir).resolve()
    expected = registry.workspace_root(run) / name
    requested = str(task.get("output_folder") or ctx.knob("output_folder", "") or "").strip()
    output = (run / requested).resolve() if requested and not Path(requested).is_absolute() else Path(requested).resolve() if requested else expected
    if output != expected:
        raise ValueError(f"portable output_folder must be the native run-local package path {expected}; external output placement is unsupported")
    if registry.where(name) is not None:
        raise ValueError(f"package {name!r} clashes with a shipped package; choose a new name")
    inputs = tuple(dict.fromkeys(p.strip() for p in str(ctx.knob("requires_input", "") or "").split(",") if p.strip()))
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", p) for p in inputs):
        raise ValueError("requires_input must list native runtime knob names")
    request = {"schema_version": 1, "name": name, "brief": brief, "variant": variant,
               "project_path": str(source), "artifact_project": str(run), "output_folder": str(output),
               "requires_input": list(inputs), "publication": "local-artifact-only"}
    previous = ctx.read_json(".factory/create-request.json")
    if previous is not None and previous != request:
        raise ValueError("run directory belongs to a different Create request; use a fresh run directory")
    _publish_template(ctx)
    ledger = S + "user-intent.md"
    if not ctx.exists(ledger):
        ctx.write_text(ledger, "# User intent\n\n## Original request (preserve every ask)\n\n" + brief + "\n\n## Authoring identity\n\n" + json.dumps(request, indent=2) + "\n")
    else:
        # Resume preserves feedback; incompatible reuse cannot silently replace intent.
        text = ctx.read_text(ledger)
        if brief not in text:
            raise ValueError("existing intent ledger does not contain this request; use a fresh run directory")
    ctx.write_json(".factory/create-request.json", request)
    if not ctx.exists(R + "repair-feedback.md"):
        ctx.write_text(R + "repair-feedback.md", "# Model-authored repair feedback\n\nNo prior repair verdict. This appendix does not alter the human request or approved strategy.\n")
    # Bootstrap is a snapshot in this run, never a mutation of the studied project.
    if source != run:
        for relative in ("factory.md", ".factory/config.json"):
            original = source / relative
            if original.is_file() and not ctx.exists(relative):
                ctx.write_text(relative, original.read_text())
    return {"create_fidelity": True, **{k: request[k] for k in ("name", "brief", "variant", "project_path", "artifact_project", "output_folder")},
            "requires_input": ",".join(inputs), "native_api": NATIVE_API}


def gate_has_factory(ctx: Any) -> str:
    return "proceed" if ctx.exists(".factory/config.json") else "halt"


def gate_factory_md_exists(ctx: Any) -> str:
    return "proceed" if ctx.exists("factory.md") else "halt"


_IGNORED = frozenset((".git", ".venv", "node_modules", "__pycache__", ".factory", ".codex", ".agents", "dist", "build"))


def _sources(root: Path):
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_symlink():
                yield path, path.relative_to(root).as_posix()


def discover(ctx: Any) -> None:
    source = Path(ctx.facts["project_path"])
    files = [rel for _, rel in _sources(source)]
    manifests = {}
    for file in ("pyproject.toml", "package.json", "AGENTS.md", "CLAUDE.md", "README.md"):
        path = source / file
        if path.is_file():
            manifests[file] = {"sha256": _sha(path.read_bytes()), "content": path.read_text(errors="replace")}
    bootstrap_context = {}
    for label, relative in (("eval_spec", ".factory/eval_spec.json"),
                            ("research_configuration", ".factory/strategy/current.md")):
        path = source / relative
        data = path.read_bytes() if path.is_file() else None
        content = data.decode() if data is not None else None
        applicable = content is not None and (label == "eval_spec" or
                     re.search(r"^#{1,6}\s+Research Configuration\s*$", content, re.MULTILINE) is not None)
        bootstrap_context[label] = {"source_path": relative, "exists": content is not None,
                                    "applicable": applicable}
        if applicable:
            bootstrap_context[label].update({"source_sha256": _sha(data), "content": content})
    ctx.write_json(".factory/eval_profile.json", {"schema_version": 1, "adapter": "native-source-discovery",
                   "project_path": str(source), "files": files, "manifests": manifests,
                   "bootstrap_context": bootstrap_context,
                   "eval_status": "not_exercised", "note": "Source inventory; evaluation commands require actual execution evidence."})


def factory_init(ctx: Any) -> None:
    data = ctx.path("factory.md").read_bytes()
    text = data.decode()
    sections = {}
    current = "preamble"
    for line in text.splitlines():
        if line.startswith("#"):
            current = line.lstrip("# ").strip()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    if not text.strip() or not any(k.lower().startswith("goal") for k in sections):
        raise ValueError("factory.md must contain a substantive Goal section")
    config = {"schema_version": 1, "adapter": "native-factory-config",
                   "source_sha256": _sha(data), "sections": {k: "\n".join(v).strip() for k, v in sections.items()},
                   "request": ctx.read_json(".factory/create-request.json")}
    if ctx.exists(FACTORY_TEMPLATE_PATH) or ctx.exists(FACTORY_TEMPLATE_SOURCE_PATH):
        canonical, provenance = _template_input()
        if (not ctx.path(FACTORY_TEMPLATE_PATH).is_file()
                or ctx.path(FACTORY_TEMPLATE_PATH).read_bytes() != canonical.encode()
                or ctx.read_json(FACTORY_TEMPLATE_SOURCE_PATH) != provenance):
            raise ValueError("bootstrap template input or provenance changed before factory initialization")
        config["bootstrap_template_input"] = provenance
    ctx.write_json(".factory/config.json", config)


def graph_update(ctx: Any) -> None:
    """Extract factual Python symbols/imports and a hashed source inventory."""
    root = Path(ctx.facts["project_path"])
    nodes, edges, inventory, errors = [], [], {}, []
    for path, relative in _sources(root):
        if path.suffix not in (".py", ".toml", ".json", ".md", ".ts", ".js", ".mjs"):
            continue
        data = path.read_bytes()
        inventory[relative] = {"sha256": _sha(data), "bytes": len(data)}
        if path.suffix != ".py":
            continue
        module = relative.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        nodes.append({"id": module, "kind": "module", "path": relative})
        try:
            tree = ast.parse(data, filename=relative)
        except (SyntaxError, ValueError) as exc:
            errors.append({"path": relative, "error": str(exc)})
            continue
        for obj in tree.body:
            if isinstance(obj, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbol = module + ":" + obj.name
                nodes.append({"id": symbol, "kind": "class" if isinstance(obj, ast.ClassDef) else "function",
                              "path": relative, "line": obj.lineno, "docstring": ast.get_docstring(obj) or ""})
                edges.append({"source": module, "target": symbol, "kind": "defines"})
            if isinstance(obj, ast.Import):
                for alias in obj.names:
                    edges.append({"source": module, "target": alias.name, "kind": "imports"})
            elif isinstance(obj, ast.ImportFrom):
                edges.append({"source": module, "target": "." * obj.level + (obj.module or ""), "kind": "imports"})
    ctx.write_json("graph.json", {"schema_version": 1, "adapter": "native-ast-knowledge-graph", "project_path": str(root),
                   "nodes": nodes, "edges": edges, "inventory": inventory, "parse_errors": errors,
                   "source_digest": _sha(json.dumps(inventory, sort_keys=True).encode())})


def study(ctx: Any) -> None:
    graph = ctx.read_json("graph.json")
    if not isinstance(graph, dict) or graph.get("adapter") != "native-ast-knowledge-graph":
        raise ValueError("study requires the native source graph")
    modules = [n for n in graph["nodes"] if n["kind"] == "module"]
    entrypoints = [n for n in graph["nodes"] if n.get("id", "").endswith(":main") or "cli" in n.get("path", "")]
    lines = ["# Native project observations", "", f"Studied project: {graph['project_path']}",
             f"Source digest: {graph['source_digest']}", "", "## Structural evidence",
             f"- {len(graph['inventory'])} hashed source/document files; {len(modules)} Python modules.",
             f"- {len(graph['edges'])} direct symbol/import relationships; {len(graph['parse_errors'])} parse failures.",
             "", "## Entry points and source locations"]
    lines.extend(f"- `{n['id']}`: `{n['path']}` line {n.get('line', 1)}" for n in entrypoints)
    lines.extend(["", "## Module inventory"])
    lines.extend(f"- `{n['id']}` — `{n['path']}`" for n in modules)
    lines.extend(["", "## Parse limitations", json.dumps(graph["parse_errors"], indent=2), "",
                  "## Interpretation boundary", "This stage extracts source evidence. The graph explorer and researchers must inspect actual code, dependencies and user intent before choosing an architecture. Test success has not been inferred from source."])
    ctx.write_text(S + "observations.md", "\n".join(lines) + "\n")


def concat_study(ctx: Any) -> None:
    parts = [ctx.read_text(S + name) for name in ("observations.md", "graph-context.md")]
    if any(len(part.strip()) < 50 for part in parts):
        raise ValueError("study concatenation requires both substantive source observations and graph exploration")
    ctx.write_text(S + "study-combined.md", "\n\n".join(parts) + "\n")


def _native_root(project: Path, name: str) -> Path:
    root = registry.workspace_root(project) / _name(name)
    if not (root / "__init__.py").is_file():
        raise ValueError(f"native package missing at {root}")
    if root.is_symlink() or not root.resolve().is_relative_to(Path(project).resolve()):
        raise ValueError("native package root escapes its artifact project")
    return root


def validate_native_package(project: Path, name: str, *, requires_input: tuple[str, ...] = ()) -> dict[str, Any]:
    """Run genuine native validation without optimization-only admission rules."""
    project = Path(project).resolve()
    problems = []
    try:
        root = _native_root(project, name)
        before = package_digest(root)
        for file in root.rglob("*.py"):
            if "__pycache__" not in file.parts:
                ast.parse(file.read_bytes(), filename=str(file))
    except (ValueError, OSError, SyntaxError) as exc:
        return {"schema_version": 1, "ok": False, "name": name, "problems": [str(exc)], "raw_evidence": {}}
    # This uses native validate_state and registry.resolve, in the same containment
    # boundary used by ordinary generated-package validation. No generated import
    # runs in the parent, and structural admission cannot author any missing code.
    code = "\n".join([
        "import importlib, json", "from pathlib import Path", "from refactory_dsh import registry",
        "from refactory_dsh.nn import validate_state", f"project = Path({str(project)!r})",
        f"package = registry.load({name!r}, project)", "problems = []", "modes = []",
        f"if package.name != {name!r}: problems.append('PACKAGE.name differs from requested name')",
        "if not package.modes: problems.append('PACKAGE contains no modes')",
        "for module in package.ops: importlib.import_module(module)",
        "for mode_class in package.modes:", "    mode = mode_class()", "    state = mode.state_dict()",
        "    issues = validate_state(state, invariants=mode.invariants(), resolve_calls=True)",
        "    problems.extend(str(issue) for issue in issues)",
        "    if not mode.is_scored() and not mode.unscored_reason: problems.append(mode.name + ': declare an honest unscored_reason')",
        f"    for wanted in {tuple(requires_input)!r}:",
        "        if wanted not in mode.parameters(): problems.append(mode.name + ': missing runtime input ' + wanted)",
        "    identifier = package.name + '/' + mode.name", "    resolved = registry.resolve(project, identifier)",
        "    modes.append({'id': identifier, 'description': mode.description, 'state': state, 'unscored_reason': mode.unscored_reason})",
        "if package.name not in registry.discover_names(project): problems.append('package is not discoverable')",
        "print(json.dumps({'problems': problems, 'modes': modes, 'describe': package.describe()}))",
    ])
    result = sandbox.run_sandboxed(code, workdir=project)
    problems.extend(result.payload.get("problems", []))
    if not result.ok:
        problems.append(result.problem() or "native validation subprocess failed")
    if not isinstance(result.payload.get("modes"), list) or not result.payload.get("modes"):
        problems.append("native validation supplied no inspectable modes")
    after = package_digest(root)
    if before != after:
        problems.append("package changed during native admission")
    return {"schema_version": 1, "ok": not problems, "name": name, "package_digest": after,
            "requires_input": list(requires_input), "problems": problems,
            "modes": result.payload.get("modes", []), "describe": result.payload.get("describe", {}),
            "raw_evidence": {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode,
                             "timed_out": result.timed_out, "timeout_s": result.timeout_s}}


def export_skill(project: Path, name: str, *, admission: dict[str, Any] | None = None) -> Path:
    admission = admission or validate_native_package(project, name)
    if not admission.get("ok"):
        raise ValueError("cannot export a skill for a package that failed native validation")
    root = _native_root(Path(project), name)
    if package_digest(root) != admission["package_digest"]:
        raise ValueError("package changed after validation; repeat admission")
    lines = ["---", "name: " + name, "description: " + str(admission["describe"].get("description", "")), "---", "",
             "# " + name, "", "Generated from the admitted native Package/Mode declarations.",
             "Package digest: `" + admission["package_digest"] + "`", "",
             "## Discovery and validation", "```sh", f"python -m refactory_dsh.registry list --project {str(project)!r} --json",
             f"python -m refactory_dsh.create_fidelity_ops validate {name} --project {str(project)!r}", "```"]
    for mode in admission["modes"]:
        state = mode["state"]
        lines.extend(["", "## " + mode["id"], "", mode["description"], "", "### Runtime parameters", ""])
        for knob, entry in state["knobs"].items():
            lines.append(f"- `{knob}` ({entry['spec']['kind']}), default `{entry['value']}`: {entry['spec'].get('description', '')}")
        lines.extend(["", "### Invocation", "```sh", f"python -m refactory_dsh.create_fidelity_ops run {mode['id']} --project {str(project)!r} --workdir NEW_RUN_DIRECTORY --knob NAME=VALUE", "```", "", "### Graph", "```json", json.dumps(state["graph"], indent=2), "```"])
    output = root / "SKILL.md"
    output.write_text("\n".join(lines) + "\n")
    return output


def admit_native_package(project: Path, name: str, *, requires_input: tuple[str, ...] = ()) -> dict[str, Any]:
    try:
        root = _native_root(Path(project), name)
        # A model-authored manifest cannot vouch for its own trusted status.
        registry._write_origin(root, origin="create_fidelity local authoring", trust="generated", source_name=name)
    except (ValueError, OSError):
        pass  # validation below retains the real missing-root diagnostics
    report = validate_native_package(project, name, requires_input=requires_input)
    if report["ok"]:
        report["skill"] = str(export_skill(project, name, admission=report))
        report["skill_sha256"] = _sha(Path(report["skill"]).read_bytes())
    root = registry.workspace_root(project) / name
    if root.is_dir():
        (root / ".native-admission.json").write_text(json.dumps(report, indent=2) + "\n")
    review = Path(project) / R / "native-admission.json"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_native_mode(project: Path, identifier: str, workdir: Path, knobs: dict[str, str]) -> dict[str, Any]:
    package_name, _ = registry.split(identifier)
    admission = validate_native_package(project, package_name)
    if not admission["ok"]:
        return {"exit_ok": False, "error": "; ".join(admission["problems"]), "raw_evidence": admission["raw_evidence"]}
    project, workdir = Path(project).resolve(), Path(workdir).resolve()
    code = "\n".join([
        "import json", "from pathlib import Path", "from refactory_dsh import registry", "from refactory_dsh.nn import execute", "from refactory_dsh.nn.edits import configure",
        f"_, mode_class = registry.resolve(Path({str(project)!r}), {identifier!r})", "mode = mode_class()",
        "state = mode.state_dict()", f"overrides = {knobs!r}",
        "for key, raw in overrides.items():", "    state = configure(state, key, raw)",
        "mode.load_state_dict(state)", f"result = execute(mode.graph(), Path({str(workdir)!r}), knobs={{key: entry['value'] for key, entry in mode.state_dict()['knobs'].items()}})",
        "print(json.dumps({'exit_ok': result.exit_ok, 'error': result.error, 'facts': result.facts, 'workdir': str(result.workdir)}))",
        "raise SystemExit(0 if result.exit_ok else 1)",
    ])
    result = sandbox.run_sandboxed(code, workdir=project)
    return {**result.payload, "exit_ok": bool(result.ok and result.payload.get("exit_ok")),
            "error": result.payload.get("error") or (result.problem() if not result.ok else None),
            "raw_evidence": {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode,
                             "timed_out": result.timed_out, "timeout_s": result.timeout_s}}


def _explicit_adverse_finding(line: str) -> bool:
    """Recognize authored adverse classifications, not words in test evidence."""
    return bool(re.match(
        r"^(?:\[(?:FAIL|NOT_VERIFIED|BLOCKER)\]\s+"
        r"|(?:status|result|verdict|adversarial verdict)\s*:\s*(?:FAIL(?:ED)?|NOT_VERIFIED)\b"
        r"|severity\s*:\s*(?:CRITICAL|HIGH|MEDIUM|LOW|BLOCKER)\b"
        r"|(?:failure|blocking finding)\s*:\s+)", line, re.IGNORECASE))


def synthesize_qa(ctx: Any) -> None:
    """Consume every director report in full, without model-created substitutes."""
    findings = ctx.read_json(".factory/qa_director-findings.json")
    body = ctx.read_text(".factory/qa_director-findings.md")
    if not isinstance(findings, dict) or findings.get("status") != "completed" or len(body.strip()) < 100:
        raise ValueError("QA synthesis requires complete director findings and provenance")
    reports = findings.get("reports")
    if not isinstance(reports, list) or len(reports) < 4:
        raise ValueError("QA synthesis requires custom testers and both mandatory testers")
    current_reports = {}
    issues = {}
    for report in reports:
        if not isinstance(report, dict) or report.get("status") != "completed" or not isinstance(report.get("body"), str):
            raise ValueError("QA synthesis found an incomplete child report")
        output = str(report.get("output", ""))
        content = report["body"]
        if not output or not ctx.exists(output) or ctx.read_text(output) != content or report.get("output_digest") != _sha(content.encode()):
            raise ValueError("QA report file, embedded body or output digest is stale")
        task_id = report.get("task_id")
        if not task_id or task_id in current_reports:
            raise ValueError("QA reports lack independent unique tester identities")
        current_reports[task_id] = report
        fenced = False
        for line in content.splitlines():
            # Preserve all original reports below. Only exact substantive issue
            # descriptions explicitly classified adverse can be promoted across
            # peers. A successful negative test often contains "failed" too.
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            cleaned = re.sub(r"^[\s#*\-\d.]+", "", line.replace("**", "")).strip()
            normalized = re.sub(r"\s+", " ", cleaned).casefold()
            adverse = _explicit_adverse_finding(cleaned)
            if len(normalized) >= 50 and (adverse or re.search(r"\b(fail\w*|missing|incorrect|violat\w*|bug|wrong|blocking)\b|does not", normalized)):
                entry = issues.setdefault(normalized, {"finding": line.strip(), "testers": set(), "adverse": adverse})
                entry["testers"].add(task_id)
    if not {"workflow-validate", "cli-integration"} <= set(current_reports):
        raise ValueError("QA synthesis is missing a mandatory native tester")
    health, code = ctx.read_text(R + "health-check.md"), ctx.read_text(R + "code-review.md")
    if min(len(health.strip()), len(code.strip())) < 50:
        raise ValueError("QA synthesis requires substantive health and code reviews")
    confidence = ["## Cross-tester finding confidence"]
    for entry in issues.values():
        testers = sorted(entry["testers"])
        if entry["adverse"]:
            label = "HIGH" if len(testers) > 1 else "SINGLE SOURCE"
        else:
            label = "REPEATED EVIDENCE" if len(testers) > 1 else "UNCLASSIFIED EVIDENCE"
        confidence.append(f"- {label} ({', '.join(testers)}): {entry['finding']}")
    if not issues:
        confidence.append("No exact substantive issue descriptions overlap; full reports remain below for CEO semantic review.")
    feedback = ctx.read_text(R + "repair-feedback.md") if ctx.exists(R + "repair-feedback.md") else "No repair feedback supplied."
    sections = ["# Synthesized Create QA", "", "## Model-authored repair feedback (complete appendix)", feedback,
                "\n".join(confidence), "## Health checker (complete report)", health,
                "## Code reviewer (complete report)", code, "## Adversarial reports and provenance (complete director publication)", body,
                "## Complete machine-readable director evidence", "```json", json.dumps(findings, indent=2), "```",
                "## Confidence policy", "Only matching substantive findings explicitly marked adverse by two or more independent testers receive HIGH confidence. Ambiguous repeated statements remain repeated evidence, including successful invalid-input probes. Single-source and unclassified evidence remain visible for CEO semantic review against actual user intent. Passing structure alone does not establish acceptance."]
    ctx.write_text(R + "qa-synthesized.md", "\n\n".join(sections) + "\n")


def gate_precheck(ctx: Any) -> str:
    """Record the honest final native check; HALT still leads to diagnostics."""
    name = str(ctx.facts["name"])
    project = Path(ctx.facts["artifact_project"])
    inputs = tuple(p for p in str(ctx.facts.get("requires_input", "")).split(",") if p)
    evidence_errors = (ValueError, TypeError, KeyError, AttributeError, OSError)
    try:
        report = validate_native_package(project, name, requires_input=inputs)
    except evidence_errors as exc:
        report = {"schema_version": 1, "name": name, "ok": False,
                  "problems": [f"native validation evidence unavailable: {type(exc).__name__}: {exc}"], "raw_evidence": {}}
    problems = list(report["problems"])
    from refactory_dsh.create_fidelity_gates import human_current_approval

    try:
        human_approved = human_current_approval(ctx)
    except evidence_errors as exc:
        human_approved = False
        problems.append(f"human approval evidence unreadable: {type(exc).__name__}: {exc}")
    if not human_approved:
        problems.append("human strategy approval is absent or stale; materially changed strategy or intent requires fresh human review")
    try:
        root = _native_root(project, name)
        receipt = ctx.read_json(R + "native-admission.json", {})
        if not isinstance(receipt, dict):
            raise ValueError("native admission receipt is not an object")
        if receipt.get("package_digest") != report.get("package_digest") or not receipt.get("ok"):
            problems.append("builder's native admission receipt is absent, failed, or stale")
        skill = root / "SKILL.md"
        if not skill.is_file() or receipt.get("skill_sha256") != _sha(skill.read_bytes()):
            problems.append("generated SKILL.md is missing or stale against native admission")
        if not (root / "README.md").is_file() or len((root / "README.md").read_bytes()) < 50:
            problems.append("requested native package README is missing or not substantive")
    except (ValueError, OSError) as exc:
        problems.append(str(exc))
    for stage in ("gate_build", "gate_qa", "gate_overwatch", "gate_doc_freshness"):
        if ctx.facts.get(stage + "_verdict") != "proceed":
            problems.append(stage + " has no current successful verdict")
        else:
            from refactory_dsh.create_fidelity_gates import current_approval

            try:
                approved = current_approval(ctx, stage)
            except evidence_errors as exc:
                approved = False
                problems.append(stage + f" approval evidence unreadable: {type(exc).__name__}: {exc}")
            if not approved:
                problems.append(stage + " approval evidence is absent or stale")
    report["ok"] = not problems
    report["problems"] = problems
    report["outcome"] = "proceed" if report["ok"] else "halt"
    ctx.write_json(R + "precheck.json", report)
    ctx.set_fact("create_fidelity_precheck_ok", report["ok"])
    ctx.set_fact("create_fidelity_precheck_error", "; ".join(problems))
    return report["outcome"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native portable Create validation, inspection, skill export and execution")
    parser.add_argument("verb", choices=("validate", "inspect", "export", "admit", "run"))
    parser.add_argument("target")
    parser.add_argument("--project", type=Path, default=project_for_nodes())
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--knob", action="append", default=[])
    parser.add_argument("--requires-input", default="")
    args = parser.parse_args(argv)
    try:
        if args.verb == "run":
            if args.workdir is None:
                parser.error("run requires --workdir")
            knobs = {}
            for entry in args.knob:
                key, separator, value = entry.partition("=")
                if not separator:
                    raise ValueError("--knob requires NAME=VALUE")
                knobs[key] = value
            report = run_native_mode(args.project, args.target, args.workdir, knobs)
            print(json.dumps(report, indent=2))
            return 0 if report["exit_ok"] else 1
        inputs = tuple(p.strip() for p in args.requires_input.split(",") if p.strip())
        report = (admit_native_package if args.verb == "admit" else validate_native_package)(args.project, args.target, requires_input=inputs)
        if args.verb == "export" and report["ok"]:
            report["skill"] = str(export_skill(args.project, args.target, admission=report))
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    except (ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
