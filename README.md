# Refactory

![Refactory — agentic harnesses, built to improve](refactory/docs/assets/refactory-hero.svg)

**Create a workflow. Run it interactively. Optimize the way it works.**

Refactory turns agentic work into reusable **modes**: explicit graphs of coding agents, deterministic operations, checks and approval gates. Drive them from a terminal or browser, steer work as it runs, and search their prompts, graph structure and parameters with a configurable outer loop.

[Get started](refactory/docs/getting-started.md) · [Terminal & GUI](refactory/docs/interfaces.md) · [Command reference](refactory/docs/cli.md) · [Write a mode](refactory/docs/tutorials/01-write-a-mode.md)

The browser keeps Refactory’s original two-gear identity and blue theme. The Modes panel has separate Run and Optimize views, Package → Mode selection, contextual help and a large brief editor. A live workflow map and required questions stay attached to the watched run. [Panel guide](refactory/docs/tutorials/07-pick-and-watch-from-the-panel.md) · [Upstream compatibility](refactory/docs/upstream-compatibility.md)


**Release status · 26 September 2026:** Profiles, live controls/recovery and DSH/Codex/Claude Code workers have completed bounded CLI and native GUI acceptance. Fast authoring is implemented, but successful completion through QA and bounded repair remains unverified. [Readiness and evidence ↓](#release-capabilities-and-validation)

## What you can build

- **A reusable harness:** Create turns a brief into a package of modes. Bind a Task to work over a class of inputs, or choose a single-run harness.
- **A better version:** Edit produces a new mode package with provenance and honest comparison status.
- **An artifact:** Design builds software, files or reports against a brief and a check command.
- **A measured improvement:** Optimize proposes allowed changes, validates them, evaluates candidates and audits the winner before adoption.

![Create and Edit produce reusable modes; Design produces artifacts; scored modes can be run, optimized and adopted](refactory/docs/assets/refactory-flow.svg)

## Why explicit graphs?

The runtime tracks which node is due, executes operations, enforces gates and records the evidence. The conversational CEO delegates agent work and accepts your steering. Skills can help a worker, but **a skill document is not the graph executor**.

The graph and knobs are inspectable data. The optimizer proposes typed changes to those data; the mode's scoring function and declared invariants remain outside that edit surface. A user-owned approval gate belongs to the user.

## Quickstart

Requires Git, [Node.js](https://nodejs.org/) `^22.19` or `>=24`, and [uv](https://docs.astral.sh/uv/). The installer sets up the pinned dependencies, Python environment and DSH profile.

```bash
git clone https://github.com/akashgit/refactory-dsh.git
cd refactory-dsh
./install.sh
uv run refactory list
```

Start with a small **offline** search over a shipped example:

```bash
uv run refactory optimize circle_packing/adaptive \
  --optimizer random --preset knob_only --evaluator walker \
  --epochs 2 --batch-size 4 --max-evaluations 8 --seed 5 \
  --runs-root .factory/loop/first-search

uv run refactory leaderboard --run .factory/loop/first-search
uv run refactory best --run .factory/loop/first-search
```

This example needs no model key. It demonstrates candidate changes, scores and a recorded stopping condition. Use a new run directory for a separate search.

Keep the winner as a new package:

```bash
uv run refactory adopt --run .factory/loop/first-search --as first_tuned
uv run refactory package smoke first_tuned/adaptive
```

The original package remains available. [Continue the walkthrough →](refactory/docs/getting-started.md)

## Choose your interface

| Terminal | Browser |
|---|---|
| `refactory ceo` for conversation and approval gates | CEO chat for authoring and node-by-node interaction |
| `refactory optimize` for explicit search configuration | Modes panel for saved profiles, knobs, search and resource limits |
| `status`, `candidate`, `steer`, `adopt` for inspection and control | Run once, live graph/worker activity, questions and candidate comparison |

```bash
uv run refactory ui --port 3081
```

For all native web presets plus CEO and the Refactory Modes panel:

```bash
./refactory-dsh --additive --port 3081
```

This creates and reuses `refactory-web`, declaring the web and Refactory bundles
in `$DSH_HOME/profiles/refactory-web/package.json`. CEO is the default, and the
existing `web` profile stays unchanged.

Open the exact URL printed by the server, select the checkout as your workspace, then create a session. Open the right sidebar's **Modes** tab; no introductory chat message is required.

Before model-backed work, save a provider key in **Settings → Models**. Nested workers and validation use the managed credential store; a checkout `.env` or exported key alone is insufficient for those scrubbed processes. `uv run refactory doctor` checks configuration, not authentication or quota. [Model setup →](refactory/docs/models.md)

Try this in GUI chat:

> Open counting_pack/fixed_step with target=20 and step=2. Drive it one node at a time and show me the recorded result.

Or from the terminal:

```bash
uv run refactory ceo counting_pack/fixed_step \
  --knob target=20 --knob step=2 --intent "Count to twenty and explain the recorded result."
```

Chat supports Create, Edit and Design. The Modes panel's **Start** button launches optimization of a scored mode; **Run once** executes any enabled mode, including full and fast authoring modes. Attach a run to follow durable activity, worker receipts and pending questions. [Interface guide →](refactory/docs/interfaces.md)

## Create, edit or design

| Goal | Example |
|---|---|
| Create a single-run harness | `uv run refactory ceo "Draft and check a project README from a supplied brief" --mode create --name readme_pack --single` |
| Create a harness for a class of inputs | Use `--task module:TaskClass` instead of `--single`; see [Tasks and datasets](refactory/docs/datasets.md) |
| Revise a mode | `uv run refactory ceo "Improve the search strategy" --mode edit --revise circle_packing/adaptive --name revised_pack` |
| Build a checked artifact | `uv run refactory ceo "Build wordcount.py and a sample.txt; verify line, word and character counts" --mode design --name wordcount --check "python wordcount.py sample.txt"` |

These commands call a model and can take substantial time. The current full authoring modes include research and reviews; you approve the strategy/design before implementation. Generated packages live in `.factory/packages/`; Design artifacts live in `.factory/artifacts/`. Exact paths and validation outcomes are recorded in the run.

For a small, focused task, choose `create_fast`, `edit_fast` or `design_fast`. Create plans a minimal reusable graph, then runs one coder and one QA agent sequentially; structural admission and canonical smoke remain required. Design uses the same three sequential roles for a tiny checked artifact, such as a one-file helper. Edit goes directly to one coder and one QA agent while preserving the source contract and optional score comparison. These graphs omit research and reviewer fan-out and allow one repair by default. All three fast paths completed their bounded CLI and native GUI checks, including real structural repairs and independent artifact checks. Prior failures are retained. See [fast modes and shared budgets](refactory/docs/budgets-fast-modes.md).

Canonical smoke/acceptance runs inside bounded repair **before** QA and Overwatch sign-off. Reviewers reuse its receipts and map the brief, accepted criteria/design and current intent ledger to actual evidence. Required behavior without evidence cannot pass; repairs and later instructions invalidate old sign-off. Full modes keep separate health, code and adversarial reviews plus Overwatch; fast modes keep one focused QA worker. [Evidence and graph details](refactory/docs/authoring-validation.md).

For an intentional Edit tradeoff, `--knob allow_regression=true` skips **all original-mode execution and numerical comparison** while still validating the revised mode and preserving provenance. The default remains false. Skipped comparison is unmeasured, never a non-regression pass.

Design produces the artifact itself. It does not need to produce a scored mode or enter the optimizer. [Authoring guide →](refactory/docs/tutorials/03-author-with-a-model.md)

## Optimize with a clear contract

Choose an outer-loop strategy, a candidate proposer, an evaluator and permitted edits. Save a named [optimization profile](refactory/docs/optimization-profiles.md), preview its objective and validation contract, and launch the same settings from CLI or GUI. Each run retains its exact resolved snapshot. Candidate comparison includes lineage, changes, raw/adjusted scores, failures, trial variability, duration and available cost/usage.

GUI chat can also launch a saved optimization profile, inspect its winner, adopt it and replay the adopted mode through Refactory tools. This path has [separate native chat acceptance](refactory/docs/release-validation-20260926.md#gui-chat-optimization-follow-up) alongside the CLI and panel checks.

During a search you can add instructions, keep or reject candidate branches, pause/resume/stop at generation boundaries, and request adoption. A failed or unaudited candidate cannot be treated as a verified winner. A run may finish without finding an improvement.

**Candidate limits, elapsed limits and token limits are separate controls.** Run-wide token/elapsed caps are optional and unset by default; fast work is bounded by its graph, repair policy and operation timeouts. [Choose a budget](refactory/docs/budget-strategy.md) using progress and measured usage rather than treating one token quota as a completion guarantee. Opt-in repeated trials and final holdout validation consume the shared run allocation; validation can have a reserved portion. Missing usage stays unavailable. Different models can change quality, cost and latency. [Optimization tutorial →](refactory/docs/tutorials/02-optimize-it.md)

## Models and coding workers

DSH remains the default worker runtime, composed with Cordis. Its providers include DeepSeek, OpenAI and Anthropic. A DSH worker using an OpenAI model is still a DSH worker. Both evaluators carry the selected routing context; per-node overrides take precedence over per-role and run settings. Inspect actual execution receipts, not just the configured selection. [Routing and worker guide →](refactory/docs/models.md)

Codex and Claude Code are explicit, optional worker backends through the installed DSH provider registry. They retain native authentication and product settings; unsupported capabilities fail before dispatch. External token-limited runs are refused because complete nested accounting is not established. Refactory still owns graph order, validation and gates.

New workers default to **`skip`**, which bypasses native permission checks and, for DSH/Codex, their sandbox restrictions. Tools have the access available to your OS account. Choose `--permission-mode default` or **Native permission policy** in the GUI to retain the backend's checks. Existing saved runs and profiles preserve their policies. [Permission details](refactory/docs/models.md#model-provider-versus-coding-agent)

## Release capabilities and validation

**All four release capabilities have completed bounded CLI and native GUI acceptance.** The checks cover the supported paths and sampled artifact contracts; they do not certify arbitrary generated projects or equal model performance.

| Capability | CLI and native GUI acceptance | Guide / issue |
|---|---|---|
| Shared resource accounting and fast Create/Edit/Design graphs | **Complete:** all three fast modes, required validation, actual bounded repairs and explicit unresolved failure | [Fast modes and budgets](refactory/docs/budgets-fast-modes.md) · [#234](https://github.com/akashgit/refactory-dsh/issues/234) |
| Live activity, questions, boundary controls and supported recovery | **Complete** | [Live control and recovery](refactory/docs/live-control.md) · [#235 closed](https://github.com/akashgit/refactory-dsh/issues/235) |
| Versioned profiles, candidate comparison, exact adoption and replay | **Complete** | [Optimization profiles](refactory/docs/optimization-profiles.md) · [#236 closed](https://github.com/akashgit/refactory-dsh/issues/236) |
| Routing and optional DSH/Codex/Claude Code workers | **Complete** | [Models and workers](refactory/docs/models.md) · [#237 closed](https://github.com/akashgit/refactory-dsh/issues/237) |

The [release validation record](refactory/docs/release-validation-20260926.md) contains separate CLI/GUI evidence and resource limits. Codex and Claude Code passed native GUI artifact checks with the new `skip` default; registered stopped-owner recovery preserved completed work in both interfaces. External whole-tree token caps and in-flight worker-session resume remain unsupported. These bounded checks do not certify generated-artifact quality, equal backend performance or a clean full-repository test suite.

## The command line

The full [command reference](refactory/docs/cli.md) covers authoring, packages, running, optimization, steering and adoption. Use `uv run refactory <verb> --help` for the installed command's options.

## Learn and contribute

| Start here | Go deeper |
|---|---|
| [Getting started](refactory/docs/getting-started.md) | [Architecture](refactory/docs/architecture.md) |
| [Concepts](refactory/docs/concepts.md) | [Python API](refactory/docs/api/index.md) |
| [Tutorials](refactory/docs/tutorials/index.md) | [Tasks and datasets](refactory/docs/datasets.md) |
| [Terminal and GUI](refactory/docs/interfaces.md) | [Verification](refactory/docs/verification.md) |
| [Models and credentials](refactory/docs/models.md) | [Contributing](refactory/docs/contributing.md) |

Refactory is under active development. Execution success, declared check results and the quality of generated content are separate evidence; inspect each when deciding whether an artifact meets your needs.

### Live help and project workers

The CEO starts with a compact operating guide and can discover current mode and
operation schemas through `refactory help`, then search/read complete shipped docs
with `refactory docs`. Mode knobs, node roles, gates, examples and supported
interface routes stay visible to the model. Builds check documentation coverage
and reject stale generated references. See [discovery and extension checklist](refactory/docs/discovery.md).

Put `refactory.workers.json` at the project root to select DSH, Codex or Claude Code
workers, including a coder-only `node_routes.emit` override. CLI, native panel,
CEO workflows and optimization resolve the same defaults; saved runs retain their
routes. `refactory workers` explains effective settings and provenance. See
[worker file examples and precedence](refactory/docs/worker-routing.md).
