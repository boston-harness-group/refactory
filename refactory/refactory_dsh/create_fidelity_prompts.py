"""Versioned Create source prompts with the reviewed native DSH mapping.

The production resource is independent of the test oracle. Source obligations
remain readable; the native mapping has precedence over framework spellings.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FACTORY_TEMPLATE_PATH = ".factory/bootstrap/factory-config-template.md"
FACTORY_TEMPLATE_SOURCE_PATH = ".factory/bootstrap/factory-config-template.source.json"
FACTORY_TEMPLATE_COMMIT = "3ab6f8f7ca792c874f2654574d7f7fd6dd18b896"
FACTORY_TEMPLATE_SOURCE = "templates/factory_config.md"
FACTORY_TEMPLATE_SHA256 = "50d8a754f707e915a0b13bd8a235d201b1930939d3b6801958dde9242e8a672e"
_FACTORY_TEMPLATE_RESOURCE = Path(__file__).with_name("data") / "create_fidelity" / "v1" / "factory-config-template.json"


def factory_template() -> dict[str, Any]:
    """Load the canonical production input and verify its pinned source bytes."""
    record = json.loads(_FACTORY_TEMPLATE_RESOURCE.read_text())
    if (not isinstance(record, dict)
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
            or record.get("source_commit") != FACTORY_TEMPLATE_COMMIT
            or record.get("source_path") != FACTORY_TEMPLATE_SOURCE
            or record.get("source_sha256") != FACTORY_TEMPLATE_SHA256
            or not isinstance(record.get("text"), str)
            or hashlib.sha256(record["text"].encode()).hexdigest() != FACTORY_TEMPLATE_SHA256):
        raise ValueError("canonical factory configuration template does not match its pinned source")
    return record


NATIVE_INSTRUCTIONS = """\
## Reviewed native mapping (takes precedence over source framework spellings)
You are operating refactory-dsh Create fidelity, authoring a reusable native
Package/Mode. Keep every substantive source obligation; translate framework
operations as follows. Do not try to run remote-factory or modify its source.

The project source for study is {{project_path}}; declared .factory artifacts
are relative to this invocation's run directory. The output project is
{{artifact_project}} and the package directory is {{output_folder}}.
The requested package name is {{name}}. The complete original request is in
.factory/strategy/user-intent.md. Runtime inputs: {{requires_input}}.

* Workflow/AgentNode/FnNode/GateNode/ForkNode/JoinNode mean nn.Mode and native
  nn.Node kinds agent/op/gate/fork/join; Workflow means nn.Graph. The public API
  is refactory_dsh.nn. Agent gate ownership and human approvals stay unchanged.
* definitions.py/primitives.py and contributed workflow study mean
  refactory_dsh/nn/{graph,mode,executor,validate}.py and refactory_packages.
  graph.json contains the native code dependency graph. Inspect its module,
  symbol and import records directly, or explore the actual project source.
* New portable workflows live at <output_folder>/__init__.py, mode.py, ops.py,
  README.md, and any requested companion CLI/tests, with PACKAGE=nn.Package
  and reusable nn.Mode classes. Calls use refactory_packages.<name>.ops:fn.
  Do not patch the framework, registry wiring, or shipped package directory.
* A portable mode receives each new input through declared runtime knobs or
  actual runtime source ports. It must support distinct invocations without
  editing its package. A Task, DataNode, numeric score or optimization knob is
  needed only if requested. An intentionally unscored mode declares an honest
  unscored_reason. Never bake the benchmark instance or expected outputs in code.
* `factory workflow validate <mode>` and validate_graph() map to
  `python -m refactory_dsh.create_fidelity_ops validate <name> --project <artifact_project>`
  which imports in a contained subprocess and runs native validate_state with
  the generated mode's own invariants. Report the actual command and output.
* `factory workflow list/show` map to `python -m refactory_dsh.registry list
  --project <artifact_project> --json` and `python -m refactory_dsh.registry show
  <name>/<mode> --project <artifact_project>` (or candidate inspect for all modes).
* Skill export maps to `python -m refactory_dsh.create_fidelity_ops export <name>
  --project <artifact_project>`. This renders SKILL.md from the real admitted
  graph and its runtime declarations inside <output_folder>/SKILL.md.
* Before publishing its builder report the builder runs candidate `admit <name>
  --project <artifact_project>`: native validation, discovery/inspection and
  skill export; write raw results and commands in builder-latest.md. This is
  structural admission, not proof of behavior. Run genuine acceptance tests.
* Native mode execution is `python -m refactory_dsh.create_fidelity_ops run
  <name>/<mode> --project <artifact_project> --workdir <new-run-dir>
  --knob input_name=value`; this calls the real native executor through a
  contained subprocess. Standalone CLI and every requested input/output
  contract must also be tested independently. Propagate failed execution.
* For authoring smoke tests and verification, allocate fresh, separate run
  directories and output paths under <artifact_project> or an actually supplied
  private TMPDIR that permits both writing and reading the results. Respect
  the invocation's filesystem boundary; do not assume global /tmp is accessible.
  This governs test outputs, not user-supplied external runtime input paths.
  Reuse the same unchanged package with each invocation's declared inputs.
* Source factory study/discover/init/precheck map to the candidate's native
  bootstrap, AST graph study, and final validation/discovery/evidence checks.
  Source eval and PR diff mean the actual local tests and local artifact diff.
* CLAUDE.md conventions map to any actual AGENTS.md, CLAUDE.md and README in
  the source project. If absent, record absence; never invent conventions.
* Commit, issue, GitHub PR, merge and deployment instructions are overridden:
  this is an artifact-only local authoring run. Do not publish or modify the
  framework. Review the real package files and preserved baseline digests.
* Child dispatch is owned by native composite directors, always through DSH.
  Plans are strict JSON; quality summaries are separate prose artifacts.
  Reports live in strategy/research/, strategy/perspectives/, reviews/testers/
  under .factory. Every report body and provenance is embedded in each stage's
  .factory/<stage>-findings.json and .md. Consume substantive reports in full.
* Archival writes remain nonblocking. The build archive must read precheck.json
  and diagnose a failed precheck, preserving failure honestly.
* Source AI gate feedback appended to the user-intent file maps to the separate
  .factory/reviews/repair-feedback.md appendix. It is model-authored critique,
  supplied in full to builder, QA and gates. It cannot edit or redefine human
  intent. Original human strategy rejection feedback remains in user-intent.md
  and follows the original re-entry through strategy and fresh human approval.
  Builder repairs preserve approval only while strategy and human intent bytes
  are unchanged. Material changes require new digest-bound human approval.

## Native package API supplied by this invocation
{{native_api}}
"""


@lru_cache(maxsize=1)
def reference() -> dict[str, Any]:
    """Read the shipping prompt resource, never the independent test fixture."""
    path = Path(__file__).with_name("data") / "create_fidelity" / "v1" / "prompts.json"
    return json.loads(path.read_text())


def role_prompt(role: str) -> str:
    prompts = reference()["role_prompts"]
    if role not in prompts:
        raise ValueError(f"no frozen Create role prompt for {role!r}")
    return str(prompts[role])


def node_prompt(node_id: str) -> str:
    return str(reference()["node_prompts"].get(node_id, ""))


def load_prompt(role: str, node_id: str = "") -> str:
    """Full inherited role and node duties, followed by explicit native rules."""
    sections = [role_prompt(role)]
    if node_id:
        sections.append("## Source stage duties\n" + node_prompt(node_id))
    sections.append(NATIVE_INSTRUCTIONS)
    if node_id == "create_factory_md":
        sections.append(f"""## Native bootstrap template assignment (this stage only)
Read `{FACTORY_TEMPLATE_PATH}` and `{FACTORY_TEMPLATE_SOURCE_PATH}` in full,
then read the declared eval_profile.json and original user-intent.md inputs.
The runtime supplied an unchanged copy of `{FACTORY_TEMPLATE_SOURCE}` at pinned
source commit `{FACTORY_TEMPLATE_COMMIT}`; its exact SHA-256 is
`{FACTORY_TEMPLATE_SHA256}`. Verify the recorded source and copy hash.
Copy that supplied template to factory.md, then fill it for this actual request.
Keep the required Goal, Scope, Guards, Eval/Command, Threshold and Smoke Test
concrete; preserve applicable template sections and document native adaptations.
The specific source-stage assignment and declared factory.md output take
precedence over the inherited CEO's general prohibition on config/non-review
writes ONLY for creating this configuration in this stage. This does not permit
application-code edits, other config writes, specialist work or later stages.
For this local artifact-only run, replace the remote Target Branch/publication
defaults with an explicit not-applicable explanation; do not create a branch,
commit or PR. Use the actual native validation and requested acceptance commands.
If the requested mode is intentionally unscored, replace the live optimizer
score command/0.8 default with honest acceptance checks and their pass/fail bar;
do not fabricate a numeric score, eval/score.py or Task. If scoring is requested,
preserve its actual requested contract instead of inventing a score or result.
Only populate Eval Spec and research sections when the supplied source context
actually activates them. eval_profile.json records any existing eval_spec.json
and current.md Research Configuration, their source bytes and hashes; absent
inputs may remain harmless commented examples or be marked not applicable.
Do not convert unused example weights/research settings into active requirements.
Smoke outputs must use freshly allocated paths within this invocation's project
or a supplied readable/writable private TMPDIR, not assumed global /tmp paths.
Leave the supplied canonical template and provenance unchanged. Deliver only
factory.md; the normal delivery receipt binds its inputs and filled output.
""")
    return "\n\n".join(sections)


# Load versioned source data once at module import. Mode.build() then assembles
# its graph from fixed in-memory instructions, without doing filesystem I/O.
reference()
