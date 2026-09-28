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
a native model. DSH also accepts `provider` and `effort`; native backends reject
those fields. Use `refactory help --topic <package/mode>` to inspect actual node
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
Switching backend clears inherited provider/model/effort and backend-specific
permissions. Shared `default` and `skip` policies retain their meaning. Invalid
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

New workers default to `skip` (native bypass permissions; Codex and DSH also
bypass their own sandbox). An explicit `permission_mode: "default"` retains native
checks. Native authentication remains with Codex/Claude Code; DSH uses its managed
credential store. See [models](models.md) for installation and authentication.
Token caps remain optional and unset; whole-tree caps are unsupported for native
backends. See [CEO discovery](discovery.md) for extending the checked help contract.
