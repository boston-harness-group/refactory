# Models

DSH is the default coding worker. Configure its model routes in `~/.dsh/settings.yaml` or **Settings → Models**. The Modes panel exposes **Worker routing and shared resources** for Run once and saved optimization profiles. Supported entry points accept `--model` and `--provider` overrides without changing saved settings. The table below explains which execution paths receive those overrides.

## Where the list comes from

The adapter's catalogue is **advisory**: it is what pickers offer, not what the route accepts. The DeepSeek route ships `deepseek-v4-flash`, `deepseek-v4-pro` and `deepseek-v4-flash-vision-exp`, and it will send any id you name — so a model that is not listed still works, it just is not offered. The settings document is hot-reloaded: nothing needs a restart.

## In the GUI

**Settings → Models.** Add a provider, paste its API key, optionally a base URL, and list the models — or press **Fetch available models** and pick from what the provider reports. **Add a custom provider** takes an id, a display name, one of three wire protocols (`openai-completions`, `openai-responses`, `anthropic-messages`) and a base URL, which is how a local server (Ollama, LM Studio, vLLM) is reached. The page writes the settings document and the credential store for you; the composer's model picker then offers the new models.

## In the terminal

The same file, by hand:

```yaml
# ~/.dsh/settings.yaml
agent-default-model:
  provider: deepseek-official
  model: deepseek-v4-pro          # the default for every run that names none

llm-pi-ai:
  providers:
    openai:
      apiKeyEnv: OPENAI_API_KEY   # the *name* of the variable, never the key
    anthropic:
      apiKeyEnv: ANTHROPIC_API_KEY
      models:
        - id: claude-sonnet-4-5
          contextWindow: 200000
    ollama:                       # a local OpenAI-compatible server
      displayName: Ollama
      api: openai-completions
      baseURL: http://127.0.0.1:11434/v1
      models:
        - id: qwen3-coder
          contextWindow: 262144
```

Keys are read from the environment (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, …, including a `.env` at the checkout root) or from the managed store at `~/.dsh/.credentials.yaml` that the Models page writes. A `llm-deepseek:` section overrides the direct DeepSeek route the same way: `models:`, `baseURL:`, `apiKeyEnv:`, `thinking:`, `reasoningEffort:`.

A local OpenAI-compatible server works through the same route, and still needs *a* key — pi-ai's OpenAI protocol asks for one, so name a dummy variable and export any value (`apiKeyEnv: LOCAL_TEST_KEY`, then `LOCAL_TEST_KEY=dummy`), or set an `Authorization` header.

## Choose a model for a run

```bash
uv run refactory ceo counting_pack/fixed_step --model deepseek-v4-pro
uv run refactory ceo "an idea" --mode create --single --model deepseek-v4-pro
uv run refactory optimize circle_packing/adaptive --provider deepseek-official --model deepseek-v4-pro
```

`--model` names the id and `--provider` the route; both fall back to `agent-default-model` in your settings, and neither writes anything back, so choosing a model for one run does not change the next. A route that does not exist is a loud failure (`NO_ADAPTER: no adapter registered for provider …`), not a quiet return to the default. The same two flags exist one level down, on `dsh --profile headless|graph`, which is what `refactory` calls.

## Model selection across execution paths

| Path | Where its model comes from |
|---|---|
| Terminal `ceo` | The selected graph-runtime provider/model, or saved defaults |
| Python Create/Edit/Design agent nodes | The supplied `DshModelClient`; its route/context is carried to supported nested validation |
| Optimization with the default `walker` evaluator | The supplied model client serves the reasoning proposer and evaluated agent nodes |
| Optimization with the separate `dsh` evaluator | The supplied client context is forwarded; explicit agent workers are dispatched through the same node bridge and route as the walker |
| GUI CEO and children | The owning session and worker configuration; inspect actual child execution when validating a route |

`refactory runtime` inspects configuration; `.refactory/workers/*.json` receipts establish the selected backend, observed runtime route, version and usage. Requested settings are separate from observed settings. Explicit `node_routes` override `role_routes`, which override the run settings; empty fields use the runtime defaults. Switching backend clears inherited provider/model/effort unless the override supplies them. A provider route declared in one DSH profile's settings is invisible to the nested headless runtime a DSH worker runs; declare such routes for workers with [`provider_config`](worker-routing.md) in `refactory.workers.json`.

A random seed does not make model outputs deterministic. Changing provider/model can change tool use, latency, cost and output quality; support for a protocol is not a performance guarantee.

## Model provider versus coding agent

DSH remains the default coding-agent runtime. It supplies the tool loop, sessions, delegation and working context; an OpenAI or Anthropic model can answer through its provider adapters. Choosing an OpenAI model does not switch the worker to Codex, and choosing a Claude model does not launch Claude Code.

Select `--worker-backend codex` or `--worker-backend claude_code` explicitly for an external node worker; `--worker-backend dsh` returns to DSH. The optional dependencies reuse DSH's named-provider registry, Codex app-server and Claude Agent SDK. They are dormant until selected and can be omitted from an installation that uses only DSH. A missing selected provider fails before execution. Native product settings and authentication remain authoritative; DSH credentials are not copied to either product.

For example, the direct interactive interface accepts:

```bash
python -m refactory_dsh.interactive counting_pack/fixed_step --worker-backend codex --model YOUR_CODEX_MODEL --headless
```

Refactory dispatches only the agent node's self-contained task, retains its declared outputs and runs graph-owned checks. The worker never owns graph order, user gates or scoring.

Each node worker receives its node identity, exact working directory, absolute
input/output paths and selected Python interpreter alongside the task and current
instructions. It uses the supplied API/reference context, performs that node's
required checks, writes its outputs and returns a brief completion response.
Delivery receipts and downstream graph operations remain runtime-owned; workers
are instructed not to rediscover that protocol from adjacent run histories or
repeat later stages. This orientation is active in both the Python executor and
native workflow path; it is guidance, not proof that a worker followed it.

New workers default to `--permission-mode skip` in the CLI, API and **Worker permission mode** control. This grants workers the access available to the OS account: DSH uses `danger-full-access` with approval policy `never`; Codex uses `dangerously-bypass-approvals-and-sandbox`; Claude Code uses `bypassPermissions`. The setting applies to each launched runtime and does not change personal product settings. Use this mode only for tasks and inputs you trust.

Choose `--permission-mode default` (**Native permission policy** in the GUI) to retain the backend's native policy, which may deny file creation. Existing saved runs and profiles preserve their permission settings; historical snapshots without a recorded policy keep the native default. A backend switch resets an inherited backend-specific policy to `skip`. Codex also accepts `never`, `approve-for-me` and its full bypass name. Claude Code also accepts `dontAsk`, `acceptEdits`, `auto`, `plan` and `bypassPermissions`. Unsupported route combinations and permissions fail before launch. External per-call provider and reasoning-effort overrides are unsupported; select those through the product's native configuration. `--effort` is supported for DSH nodes.

| Adapter capability | DSH node | Codex node | Claude Code node |
|---|---|---|---|
| Model selection | Yes | Yes | Yes |
| Cancellation/deadline | Owned runtime shutdown | Owned runtime shutdown | Owned runtime shutdown |
| Steering | Next node boundary | Next node boundary | Next node boundary |
| Resume in-flight worker / images / human questions | Not exposed by this one-shot bridge | Unsupported | Unsupported |
| Observed route | DSH model request | Native thread startup and reroute events | Native SDK initialization and observed model list |
| Token reporting | Measured request totals, including nested DSH agents | Measured native thread totals; nested coverage not established | Estimated SDK query totals, including Task workers; excludes helper calls outside that pipeline |

External workers refuse token-limited launches because complete nested accounting is not established. They support elapsed limits. Missing usage stays unavailable, never measured zero. Codex/Claude product sessions are ephemeral: receipts retain their identity for provenance, but never offer resume. A failed or cancelled worker is not automatically replayed. New instructions are delivered at an explicitly chosen node boundary; an already executing external worker cannot receive an in-flight edit. The native panel disables unsupported provider/effort/token controls for external backends and displays these limits. Switching backend clears incompatible inherited values; explicit unsupported overrides still fail validation.

Observe-only patches on the pinned DSH providers emit whitelisted route/usage metadata through Cordis. They do not copy raw protocol traffic, prompts, commands, paths, reasoning or credentials into the parent conversation. The protocol and product lifecycle remain owned by the existing provider bundles. Official integration contracts: [Codex App Server](https://learn.chatgpt.com/docs/app-server), [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk), [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview).

Worker receipts also feed [live activity](live-control.md); saved [profiles](optimization-profiles.md) retain requested routes in their exact launch snapshot. Configuration readiness is not execution evidence. DSH completed the bounded CLI and native GUI task. Codex and Claude Code completed the CLI task with explicit edit permissions and the native GUI task with the new `skip` default. Earlier native-policy failures remain documented as permission controls; see the [validation record](release-validation-20260926.md).

## Credentials for nested execution

Create/edit smoke and panel optimization launch further DSH processes. Their environment
is deliberately scrubbed, and the smoke process has a workspace-local `HOME`. Save the
provider key once through **Settings → Models** in the DSH home used by the run:
open the provider row with **Edit**, enter its **API key**, and choose **Apply**. The
product passes that nonsecret home location to nested model processes; keys remain in
DSH's managed store and are resolved by its credential service. Do not copy `.env`
files into generated packages or smoke directories. If an exported key shadows the
reference, the UI reports “Provided by the launch environment (read-only)”; launch
without that override before saving. DSH resolves keys in this order: inherited launch
environment, managed store, invocation-directory `.env`, then DSH-home `.env`. A stored
key overrides both `.env` layers, but not an inherited environment override. The launch
environment is a startup snapshot; changing the launching shell afterward does not
change the running GUI's credential source.

`refactory doctor --provider deepseek-official --model deepseek-flash` checks local
credential configuration through the same scrubbed subprocess boundary as smoke. Create
and edit perform this check before research. The check makes no model request: a
configured key is not proof of valid authentication, available quota, or network access.
The later real model smoke remains required. Provider-native authentication without an
API-key reference is reported as `native-auth-unverified`, not as a verified credential.

The local diagnostic uses the installed headless profile's supported file-backed
credential/settings services and adapter schemas. A replaced service or dynamic `!!js`
configuration, an external `DSH_CLI`, or a source-checkout launcher is reported as
`diagnostic-unavailable`: the doctor prints **WARN** and authoring continues with an
explicit unverified-readiness warning. The actual selected runtime performs its normal
checks. Confirmed missing references and malformed known credential/settings configuration
still stop authoring before research. The diagnostic never guesses a credential location.
DSH's normal profile loader may initialize a missing shipped headless profile; the check
makes no model request and does not write credential values. The local managed store is owner-only but does not
isolate credentials from generated code running as the same OS user; stricter isolation
requires a separate model-service boundary.

For a DSH source checkout, set `DSH_ROOT` to its root after installing its dependencies.
The source launcher resolves `tsx` and the checkout's TypeScript configuration there;
agent processes still run in their item directories. `DSH_CLI` continues to select an
external executable directly. A custom/source runtime must support the flags supplied
to it: upstream headless profiles without this fork's `--provider`/`--model` extension
can select their route through the profile's `agent-default-model` configuration.

## Inspect context before an interactive workflow

The CEO can use the existing `refactory` tool with `action: "runtime"` before `wf_open`. It reads the current session selection and the host-owned DSH home, then uses the same nested credential diagnostic as create/edit. It never asks you to copy credentials into a run or change a configured home. Missing and invalid configuration are reported as such; unsupported diagnostics stay unverified. Current-session reasoning effort is shown separately because credential readiness does not prove nested reasoning selection.

From the terminal, `refactory runtime` reports nested defaults as JSON; `--provider` and `--model` inspect an explicit route without changing settings. There is no current CEO session in that CLI call. The diagnostic does not authenticate or launch a mode. As with doctor, the normal DSH loader may initialize a missing shipped headless profile, but it does not write credentials.

## Completion deadlines and interruption

Worker contexts retain the selected Python environment (`VENV_BIN`, defaulting to
the checkout's `.venv/bin`). Tool commands receive it first on `PATH` and receive
the installed Refactory API on `PYTHONPATH`, even when their working directory is
a run's artifact directory. DSH's normal credential scrub still applies to child
processes; selecting a Python environment does not forward credential variables.

Token-limited DSH runs require the shipped in-process `spawn` and `fork` provider
composition. An additional named provider with unverified nested telemetry is
refused before a model request. Without a token limit, native descendants may run,
but their observations remain separate from the DSH parent's route: known DSH
tokens stay visible and total usage is explicitly unavailable. A completed DSH
request without valid usage likewise prevents claiming a measured total.

A `DshModelClient` deadline fails the completion without replaying its prompt: the
agent may already have written files or started tool work. Explicit transport
failures retain their bounded retry policy. On POSIX, a deadline or caller
`KeyboardInterrupt` raised in the thread calling `complete()` sends one SIGTERM
to the selected CLI and allows ten seconds for shutdown. The Python walker's
concurrent fork also owns a cooperative cancellation scope: a main-thread
interrupt stops queued branches and asks active shipped DSH clients to shut down
before propagating that original interruption. Cancelled branches are recorded at
the fork, without merging their facts or acknowledging its join. Each fork binds
an independent client copy; cancellation does not change the original client or
other runs. Completion polling retains one absolute deadline and accumulated
output, and cancellation also stops retry launches and backoff waits. The shipped CLI disposes its context and managed subprocesses,
including ordinary tool jobs launched in separate process groups. A separate CLI
session prevents terminal Ctrl-C from also delivering a second shutdown signal.
Valid work that finishes inside the deadline follows normal CLI disposal.

This is runtime-owned cleanup, not a scan or kill of process names. A runtime that
does not settle within the grace is forced to exit; the error says child cleanup
is unconfirmed and no retry follows. Custom launchers must implement their own
shutdown, and Windows termination does not provide the POSIX graceful signal
path. The local provider's macOS process-group fallback cannot contain arbitrary
descendants that escape its managed range. Python SIGKILL, abrupt parent death,
still need external supervision. Default process-wide SIGTERM now unwinds the owned Python model subprocess before exit.
Arbitrary in-flight Python operations are awaited, not forcibly interrupted;
custom model clients retain caller-thread serial execution and their own cleanup
contract. The client does not undo files written before failure or certify their reuse.

## Project defaults and discovery

[Worker routing](worker-routing.md) documents `refactory.workers.json`, coder-only
selection and precedence. [Live help and docs](discovery.md) provide current-build
read-only discovery in chat and the terminal without source inspection.
