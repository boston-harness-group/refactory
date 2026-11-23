# Project worker routing

Put **`refactory.workers.json`** in the project root used by Refactory (the console
`--project`, or `REFACTORY_PROJECT` for the terminal). A missing file preserves
DSH defaults. The file uses the existing worker route schema; it contains no
credentials. It configures graph workers, not the CEO conversation's model.

Select Codex for all workers:

```json
{"worker_backend":"codex","permission_mode":"skip"}
```

Select Claude Code for all workers:

```json
{"worker_backend":"claude_code","permission_mode":"skip"}
```

Keep other workers on DSH and select Codex only for the shipped authoring coder:

```json
{
  "worker_backend": "dsh",
  "node_routes": {"emit": {"worker_backend": "codex"}}
}
```

Replace `codex` with `claude_code` for a Claude Code coder. Optional `model` selects
a native model. DSH also accepts `provider` and `effort`; `claude_code` accepts
`effort` but not `provider`, and Codex rejects both. `claude_code` alone accepts
`setting_sources`: `none` or a comma list of `user`, `project` and `local`, the
Claude Code settings it loads (as `claude --setting-sources`), at the run level of
the file or a route map, in an optimization profile's `setting_sources`, or with
`--setting-sources` on `refactory ceo`. Unset, it loads all of
them, or none for a node with `tool_policy="none"` or `"refused"`; set, it replaces
either default, and such a node still gets no tools (a `refused` node loads none, whatever
the route says). The worker record's `actual_route` shows the
effort and settings sources Claude Code was started with, and `enforced_tool_policy`
is set only once the runtime has applied the policy.

`claude_code` alone also accepts `max_thinking_tokens`, a positive whole number, passed
to Claude Code as its thinking budget (`--max-thinking-tokens`, as a harness passing the
flag does). What Claude Code does with it depends on its version and the model: on 2.1.299
with `claude-opus-4-6`, a `claude -p --max-thinking-tokens 16384 --effort medium` and a
worker session with this field both send adaptive thinking with effort `medium` (seen
against a stub API), so the flag has no effect on that model in either. Unset, Claude Code
chooses. A mode may bind it to a frozen knob of its own on one step's route
(`field_bindings` `("route.max_thinking_tokens", knob)`, `Node.bound_route`), which both
runtimes then carry; `spreadsheetbench/agent` does. `actual_route` records it.

`claude_code` alone also accepts `max_turns`, a positive whole number, passed to Claude
Code as its turn cap (`--max-turns`). A session that reaches it ends there, and the step's
answer is a fixed note saying so (`(Claude Code stopped the session at its turn limit of N
turns.)`, worker metadata `turn_limit_reached`), so whatever the session left in its
workspace is still collected; without the field, a session Claude Code stops at a limit
fails the step. A mode binds it as it binds the thinking budget (`("route.max_turns",
knob)`); `spreadsheetbench_v2/swe` does (50). `actual_route` records it.

`claude_code` alone also accepts `bash_timeout_ms` and `bash_output_chars`, positive whole
numbers: each Bash command's default and longest timeout (Claude Code's
`BASH_DEFAULT_TIMEOUT_MS` and `BASH_MAX_TIMEOUT_MS`), and the most characters of a
command's output the model sees inline (the `bashOutputMaxChars` flag setting, which Claude
Code clamps to 4,000-128,000, and `BASH_MAX_OUTPUT_LENGTH`; the rest is saved to a file
Claude Code names). Bound like `max_turns`; `spreadsheetbench_v2/swe` binds 60000 and 10000.
A sealed session also leaves its result's facts in its workspace,
`.refactory-session.json` (`subtype`, `num_turns`, `max_turns`, `turn_limit_reached`, and
`tool_calls`: each call's tool, its input as the model wrote it and its output, each cut to
2,000 characters, and whether it was an error), which a step collects by declaring it as a
write; nothing else reads it.

`claude_code` alone also accepts `claude_executable`, the absolute path of the Claude
Code it runs instead of the one bundled with the pinned Agent SDK (Claude Code
2.1.281). Each model needs a recent enough one: the Claude 5 family (`claude-opus-5-*`,
`claude-sonnet-5-*`, ...) needs 2.1.280 or newer. Set it in a route map, at the run
level of the file, with `--claude-executable` on `refactory ceo`, the walker's mains
and `refactory optimize`, or with `REFACTORY_CLAUDE_EXECUTABLE` (a leading `~` is
expanded). Node beats role beats run flag or file beats the variable beats the
bundled one. `actual_route` records the executable. A run whose claude_code route cannot run its
model is refused at launch, before its run directory or worktrees exist: a chosen
executable that is missing, not executable or reports no `--version`, or a version
older than the model needs; the refusal names the executable, its version, the model
and these settings. `refactory doctor` shows which Claude Code the worker runs, its
version and where the choice came from.

A claude_code step waits for the background subagents and tasks it started before it
finishes: a turn that ends while one still runs is not the step's answer, and Claude
Code carries on as each reports back. Background tasks stay on, so parallel subagents
keep their speed, and the worker record's `background_tasks_waited` counts them. A
task that reports nothing for `REFACTORY_BACKGROUND_QUIET_S` seconds (default 600), such
as a server or watcher, no longer holds the step: it ends with its last answer, Claude
Code stops what is left, and `background_tasks_abandoned` counts it. A Claude Code that
stays alive and silent after that is stopped within 30 s, and the step still finishes
with its last answer (`background_stream_stopped`). A long
research step has no wall to outgrow: it ends on inactivity (`REFACTORY_MODEL_INACTIVITY_S`).
A turn that ends with no text (often the last one, after a background report) is not
the answer either: the step asks Claude Code once, in the same session, for its complete
final answer (`empty_answer_reasked`), bounded like a silent task. If that turn is empty
too, it keeps an earlier answer only when that one was given with no task running
(`answer_from_earlier_turn`); text written while tasks still ran is an interim note, so
the step fails "the answer was empty" and the error quotes that note. A reply to the
re-ask never replaces such an earlier answer. The node is not rerun, since a session
with tools is not safe to replay.

A claude_code worker always compacts its conversation as it fills, as an interactive
Claude Code does by default, so a long step (a builder writing a large change) does not
end with "Prompt is too long" and lose its work. The adapter starts Claude Code with the
flag setting `autoCompactEnabled: true`, which outranks every other source, and does not
pass `DISABLE_AUTO_COMPACT`, `DISABLE_COMPACT` or `DISABLE_MICROCOMPACT` from the
environment that started the run (a Claude Code session sets one). The person's own
choice is untouched: an `autoCompactEnabled: false` in `~/.claude.json`, which a worker
would otherwise read through its real HOME, still applies to their own sessions and is
never written. There is no opt-out, since an unattended worker that cannot compact can
only fail once it outgrows its window. The worker record's `auto_compact` says it was on
and which switches were dropped, `compactions` counts the times Claude Code compacted in
the step, and `refactory doctor` has a `claude_code compaction` row.

Use `refactory help --topic <package/mode>` to inspect actual node
IDs and roles; custom modes may not call their coder `emit`. `role_routes` uses
the same objects keyed by declared worker role. Node beats role beats run.

A DSH worker runs a nested `dsh --profile headless` runtime, which cannot see
provider profiles declared in another profile's patch (a gateway configured only
on the web profile, say). Declare such provider routes with **`provider_config`**,
an object keyed by provider name whose values are the nonsecret profiles the
nested runtime's `llm-pi-ai` adapter accepts — endpoint, protocol, model catalog.
Credentials are never values here: name the native-store reference with
`apiKeyEnv`, exactly as the session's own provider configuration does.

```json
{
  "worker_backend": "dsh",
  "provider": "litellm",
  "model": "rits/zai-org/glm-5-3",
  "provider_config": {
    "litellm": {
      "apiKeyEnv": "LITELLM_API_KEY",
      "api": "anthropic-messages",
      "baseURL": "https://gateway.example/v1",
      "models": [{"id": "rits/zai-org/glm-5-3", "name": "rits/zai-org/glm-5-3"}]
    }
  }
}
```

`provider_config` follows the route-map replacement rule: an explicit empty object
clears it, and a saved run or inherited child context keeps the map it was created
with. The same field is accepted by CLI `open --provider-config`, and a GUI chat
session passes its own selected provider/model to a run it opens, so the run's
workers inherit the conversation's route instead of the nested runtime's default.

For a new run the precedence is built-in defaults, environment selection, project
file, selected immutable profile, then explicit launch options. Route maps replace
the lower map as a whole, so explicit `{}` clears project role/node overrides.
Switching backend clears inherited provider/model/effort, `setting_sources`, `claude_executable` and
backend-specific permissions. Shared `default` and `skip` policies retain their meaning. Invalid
fields, unsupported native provider/effort, unsupported permissions, and missing
selected provider packages fail instead of falling back to DSH.

CLI Run/authoring, panel Run once, CEO `wf_open` and optimization resolve at shared
preparation using the project root, before workers enter artifact directories.
An inherited resolved child context and a saved run snapshot bypass the file.
Edits affect new runs only; recovery keeps its saved identity and fingerprint
checks. Existing profiles remain immutable, including explicit native policies.
A parser or panel fallback is omitted, so it cannot erase a project choice.
Choose **Use project defaults** in the panel to remove an explicit backend.

Inspect without running a worker:

```sh
refactory workers --project /path/to/project
refactory profile-resolve counting_pack/fixed_step --project /path/to/project
```

The worker view reports the file path, content hash, effective values and their
sources. `.refactory/run-config.json` records the resolved context and provenance
for Run/authoring; `resolved-profile.json` records it for optimization. Worker
receipts under `.refactory/workers` record the requested and actual route. A
preview is configuration evidence; the actual receipt establishes dispatch.

### Example: routing design_doc to a stronger model

```json
{
  "node_routes": {
    "design_doc": {"model": "claude-opus-4-20250514"}
  }
}
```

This routes the design writer to a stronger model. The worker
receipt records `requested_route` and `actual_route`; the run report and PR
body's Verification section show which model planned each step.

New workers default to `skip` (native bypass permissions; Codex and DSH also
bypass their own sandbox). An explicit `permission_mode: "default"` retains native
checks. Native authentication remains with Codex/Claude Code; DSH uses its managed
credential store. See [models](models.md) for installation and authentication.
Token caps remain optional and unset; whole-tree caps are unsupported for native
backends. See [CEO discovery](discovery.md) for extending the checked help contract.
