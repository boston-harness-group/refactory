/**
 * The console process boundary. One call per console verb: build argv, run
 * `python -m refactory_dsh.console`, and return the JSON it printed.
 *
 * The loop is Python and stays there. This is the only place the tool layer
 * touches it, so the tool and the client card both read one JSON contract —
 * `refactory_dsh/console.py` — and neither reimplements the loop's file layout.
 * @module @deepseek-ai/dsh-tool-refactory/console
 */

import { join, resolve } from 'node:path'
import { homedir } from 'node:os'
import type { SubprocessHandle } from '@deepseek-ai/dsh-subprocess'
import type { SubprocessRuntime } from '@deepseek-ai/dsh-subprocess'

/** How one console invocation is addressed. */
export interface ConsoleInvocation {
  /** Console verb, e.g. `status`. */
  verb: string
  /** Positional arguments after the verb, e.g. a mode id. */
  positional?: readonly string[]
  /**
   * Long options, already spelled without the leading `--`. An array is one flag per
   * element, which is how a repeatable option (`--verdict`) is passed.
   */
  options?: Readonly<Record<string, string | number | boolean | readonly string[] | undefined>>
}

/** Resolved deployment settings for reaching the console. */
export interface ConsoleSettings {
  /** Repository root: the checkout holding `refactory/` and `.venv/`. */
  repoRoot: string
  /** Directory holding the interpreter that has refactory_dsh installed. */
  pythonBin: string
  /** Wall-clock ceiling for one console call. */
  timeoutMs: number
  /** Per-stream collection cap; a longer stream keeps its tail. */
  collectBytes: number
}

/**
 * What a console call produced: the decoded JSON, or the reason there is none.
 * `argv` is the exact command, so a failure can be reproduced by hand.
 *
 * A failed call keeps `value` when the console printed JSON: its refusal is
 * written beside the payload it is about (`console run` reports where the run
 * stopped in the same object as the reason it did not finish), and a caller that
 * needs that payload cannot read it back from the error line.
 *
 * `truncation` and `suppressed` are the two ways a call can need more than the
 * plain answer says. They are separate members rather than a richer `error`
 * string so a caller can branch on the cause instead of reading prose: the
 * collection cap cutting the payload is a size problem with a fix
 * (`collectBytes`), and a console that refused the paging options is an age
 * problem with an answer (the unpaged read).
 */
export type ConsoleResult =
  | { ok: true; value: unknown; argv: string[]; suppressed?: readonly string[] }
  | {
    ok: false
    error: string
    argv: string[]
    value?: unknown
    /** Present when the collection cap, not the console, cut the payload. */
    truncation?: {
      readonly capBytes: number
      /** Bytes actually retained in the in-memory tail. */
      readonly retainedBytes: number
    }
    /** Present when the call succeeded only by dropping options this console refuses. */
    suppressed?: readonly string[]
  }

/**
 * The options an older console refuses, long option names spelled as argv does.
 *
 * Paging the journal is the only contract that needs a console new enough to
 * serve it, so these are exactly the two flags a caller may be asked to drop.
 */
export const ACTIVITY_OPTIONS = ['activity-after', 'activity-limit'] as const

/**
 * Deployment settings for reaching the console, before they are resolved.
 *
 * Both console consumers — the model-facing tool and the Web Remote service —
 * declare these same four fields, so a deployment edits one row per consumer and
 * the shared resolver turns both into the same absolute paths.
 */
export interface ConsoleConfig {
  /** Repository root holding `refactory/` and `.venv/` (default: the process cwd). */
  readonly repoRoot?: string
  /** Directory of the interpreter with refactory_dsh installed (default `<repoRoot>/.venv/bin`). */
  readonly pythonBin?: string
  /** Wall-clock ceiling for one console call (default 10 minutes). */
  readonly timeoutMs?: number
  /** Per-stream collection cap in bytes (default 4 MiB). */
  readonly collectBytes?: number
}

/**
 * The console's deployment defaults, in one place.
 *
 * Schemastery has no non-mutating object-schema composition (its `set` edits the
 * schema in place), so each consumer declares its own fields and takes every
 * default from here: a cap can only change in one place.
 */
export const CONSOLE_DEFAULTS = {
  /**
   * Wall-clock ceiling for one console call.
   *
   * 90 minutes because the authoring verbs are the long ones, and because the
   * number this replaced was measured rather than guessed: a real creation on this
   * checkout took 48 minutes (37 inner runs, one overwatch reloop that sent the
   * build back to `emit`), which a 10-minute ceiling killed mid-flight and a
   * 45-minute one would have killed three minutes from the end. A quick verb
   * returns in seconds, so the ceiling only decides how long a hung call may hang.
   */
  timeoutMs: 90 * 60 * 1000,
  /** Per-stream collection cap in bytes. */
  collectBytes: 4 * 1024 * 1024,
} as const

/**
 * Resolve deployment settings into the absolute paths one `Console` needs.
 *
 * `pythonBin` defaults against the resolved repository root rather than in the
 * schema, because a schema default cannot read a sibling field's value.
 * @param config - loader-validated settings.
 * @returns the settings as the process boundary consumes them.
 */
export function resolveConsoleSettings(config: Required<ConsoleConfig>): ConsoleSettings {
  const repoRoot = resolve(config.repoRoot)
  return {
    repoRoot,
    pythonBin: resolve(config.pythonBin || join(repoRoot, '.venv', 'bin')),
    timeoutMs: config.timeoutMs,
    collectBytes: config.collectBytes,
  }
}

/** What one `start` call carries: the mode, where the run lives, and its settings. */
export interface StartRequest {
  /** Mode to run, as `<package>/<name>`. */
  readonly mode: string
  /** Run id to claim; the console derives one when absent. */
  readonly runId?: string | undefined
  /** Directory holding the runs; the console default when absent. */
  readonly runsRoot?: string | undefined
  /** Run settings, passed through as `--<name> <value>` flags; `undefined` is skipped. */
  readonly settings?: Readonly<Record<string, string | number | boolean | undefined>> | undefined
  /** Knob values the search starts from, as `NAME=VALUE`; one `--knob` per element. */
  readonly knobs?: readonly string[] | undefined
}

/**
 * The console invocation that starts one run. The flags are the console's, so a
 * setting the console adds is usable here without an edit.
 * @param request - the mode, run addressing, and settings.
 * @returns the verb, positionals, and options for one console call.
 */
export function startInvocation(request: StartRequest): ConsoleInvocation {
  return {
    verb: 'start',
    positional: [request.mode],
    options: {
      'run-id': request.runId,
      'runs-root': request.runsRoot,
      knob: request.knobs,
      ...request.settings,
    },
  }
}

/** What one `run` call carries: a mode used once, and the knobs it uses. */
export interface SingleRunRequest {
  /** Explicit worker route and shared resource limits. */
  readonly settings?: Readonly<Record<string, string | number>> | undefined
  /** Host-selected directory for this invocation; absent keeps the CLI default. */
  readonly workdir?: string | undefined
  /** Mode to run, as `<package>/<name>`. */
  readonly mode: string
  /** Knob values the run uses, as `NAME=VALUE`; one `--knob` per element. */
  readonly knobs?: readonly string[] | undefined
}

/**
 * The console invocation that uses one mode once.
 *
 * `run` is the CLI's run — a person using a mode — where `startInvocation` is the
 * outer loop searching one. They take the same `--knob` spelling, so the flag has
 * one owner here rather than one per caller.
 * @param request - the mode and the knob values it uses.
 * @returns the verb, positionals, and options for one console call.
 */
export function runInvocation(request: SingleRunRequest): ConsoleInvocation {
  const routeKeys = ['worker_backend', 'provider', 'model', 'effort', 'permission_mode', 'role_routes', 'node_routes',
    'max_elapsed_s', 'max_tokens', 'validation_reserve_s', 'validation_reserve_tokens']
  const settings = Object.fromEntries(Object.entries(request.settings ?? {}).filter(([key]) => routeKeys.includes(key)))
  return {
    verb: 'run',
    positional: [request.mode],
    options: { ...settings, knob: request.knobs, workdir: request.workdir },
  }
}

/**
 * The console invocation for one package-registry verb.
 * @param action - the registry verb, e.g. `list` or `enable`.
 * @param target - the package or directory it acts on; empty addresses the catalogue.
 * @param args - extra registry flags, as a shell-quoted string.
 * @returns the verb, positionals, and options for one console call.
 */
export function registryInvocation(action: string, target: string, args?: string): ConsoleInvocation {
  return { verb: 'registry', positional: [action, target], options: { args } }
}

/** One console call against the real process. */
export class Console {
  constructor(
    private readonly subprocess: SubprocessRuntime,
    private readonly settings: ConsoleSettings,
  ) {}

  /**
   * Build the argv for one invocation.
   * @param invocation - verb, positionals, and options.
   * @returns the argv, starting with the interpreter.
   */
  argv({ verb, positional = [], options = {} }: ConsoleInvocation): string[] {
    const argv = [join(this.settings.pythonBin, 'python'), '-m', 'refactory_dsh.console', verb]
    for (const [name, value] of Object.entries(options)) {
      if (value === undefined || value === false) continue
      const flag = `--${name.replaceAll('_', '-')}`
      if (value === true) argv.push(flag)
      // Forwarded registry flags are a value, even when they begin with `--`.
      else if (name === 'args') argv.push(`${flag}=${String(value)}`)
      else if (Array.isArray(value)) {
        for (const element of value) argv.push(flag, String(element))
      } else argv.push(flag, String(value))
    }
    argv.push(...positional)
    return argv
  }

  /**
   * Run one invocation and decode its JSON.
   *
   * Two verdicts are made here rather than in the caller, because both are facts
   * only this seam holds:
   *
   * - **The collected payload was truncated.** When the reader is `lossy` on the
   *   after-settlement `readFrom(0)` — the same read the JSON is decoded from —
   *   the collector dropped the head of stdout at its cap, and the retained tail
   *   is all there is. A payload that does not decode *because of that* is
   *   reported as the size problem it is, with the cap and the retained length,
   *   and never as `the console printed no JSON` — which is reserved for a
   *   complete stdout that is not JSON, including a child's own output printed
   *   ahead of the envelope (issue 95). The seam exposes no whole-stream byte
   *   total, so none is claimed; spilling is not configured for these streams,
   *   so `spillPath` is never a recovery route here.
   * - **The console refused the paging options.** A console older than the
   *   activity paging contract exits 2 with `unrecognized arguments`. That
   *   refusal is retried exactly once without the paging options, and the answer
   *   names what was dropped, so the run stays readable rather than the person
   *   being told their run is broken. Only that exact refusal is retried.
   * @param invocation - verb, positionals, and options.
   * @param signal - caller's abort signal, forwarded to the child.
   * @returns the decoded result; a nonzero exit or undecodable stdout is reported, not thrown.
   */
  async run(invocation: ConsoleInvocation, signal?: AbortSignal): Promise<ConsoleResult> {
    const result = await this.execute(invocation, signal)
    if (result.ok || result.truncation !== undefined) return result
    const suppressed = this.refusedActivityOptions(result.error, invocation, result.argv)
    if (suppressed === undefined) return result
    const fallback = { ...invocation, options: withoutActivityOptions(invocation.options ?? {}, suppressed) }
    const retried = await this.execute(fallback, signal)
    return { ...retried, suppressed: [...suppressed, ...retried.suppressed ?? []] }
  }

  /**
   * Which paging options a console refused, read from its own refusal.
   *
   * `unrecognized arguments` is argparse's exit-2 wording for an unknown option,
   * and the error line here is the console's, so the flags it names are the flags
   * it does not know. A call that sent none of them has nothing to drop, and a
   * refusal that is not that exact wording is a different failure.
   * @param error - the failure line the failed call reported.
   * @param invocation - the call that produced it.
   * @param argv - the argv the console was given.
   * @returns the refused option names, or `undefined` when nothing may be dropped.
   */
  private refusedActivityOptions(
    error: string,
    invocation: ConsoleInvocation,
    argv: readonly string[],
  ): readonly string[] | undefined {
    if (!UNRECOGNIZED.test(error)) return undefined
    const sent = new Set(Object.keys(invocation.options ?? {}))
    const refused = ACTIVITY_OPTIONS.filter(option => sent.has(option) && argv.includes(`--${option}`))
    return refused.length === 0 ? undefined : refused
  }

  /**
   * Run one invocation once, and decode what it printed.
   * @param invocation - verb, positionals, and options.
   * @param signal - caller's abort signal, forwarded to the child.
   * @returns the decoded result of that one process.
   */
  private async execute(invocation: ConsoleInvocation, signal?: AbortSignal): Promise<ConsoleResult> {
    const argv = this.argv(invocation)
    const timer = AbortSignal.timeout(this.settings.timeoutMs)
    const abort = signal === undefined ? timer : AbortSignal.any([signal, timer])
    const handle: SubprocessHandle = this.subprocess.spawn({
      argv,
      cwd: this.settings.repoRoot,
      // The provider intentionally scrubs DSH_* and secret variables. Carry only
      // the trusted location so nested DSH reads its managed credential store.
      env: { DSH_HOME: resolve((process.env.DSH_HOME?.trim() || join(homedir(), '.dsh')).replace(/^~(?=[/\\]|$)/, homedir())) },
      stdio: {
        stdin: 'ignore',
        stdout: { maxBytes: this.settings.collectBytes },
        stderr: { maxBytes: this.settings.collectBytes },
      },
      graceMs: 2_000,
      signal: abort,
    })
    const outcome = await handle.done
    const stdout = handle.collected.stdout
    const stderr = handle.collected.stderr
    if (stdout === undefined || stderr === undefined) {
      // Both were requested as collect above, so a missing reader is a provider
      // that did not honour its own spec. Report it rather than read undefined.
      return { ok: false, argv, error: 'the subprocess provider returned no collected output' }
    }
    const read = stdout.readFrom(0)
    const stdoutText = read.text
    const stderrText = stderr.readFrom(0).text
    if (outcome.exitCode !== 0) {
      // The console prints its own `{"error": ...}` on a handled refusal, so a
      // failed call still carries the reason the user needs — and the rest of the
      // payload it wrote that reason beside.
      const reported = decode(stdoutText)
      const decoded = reported.kind === 'json' ? reported.value : undefined
      return {
        ok: false,
        argv,
        ...decoded === undefined ? {} : { value: decoded },
        error: reported.kind === 'json' && isRecord(decoded) && typeof decoded.error === 'string'
          ? decoded.error
          : stderrText.trim() || `exited with ${outcome.exitCode ?? outcome.signal ?? 'no status'}`,
      }
    }
    const decoded = decode(stdoutText)
    if (decoded.kind === 'json') return { ok: true, argv, value: decoded.value }
    if (read.lossy) {
      // The cap, not the console, decided what this process printed: the head is
      // gone, so the tail cannot be decoded and no byte total can be named.
      return {
        ok: false,
        argv,
        error: `the console's output was truncated at the ${this.settings.collectBytes}-byte collection cap `
          + `before it printed JSON (only ${stdoutText.length} bytes were retained); `
          + `raise collectBytes or page the read\n${argv.join(' ')}`,
        truncation: { capBytes: this.settings.collectBytes, retainedBytes: stdoutText.length },
      }
    }
    return { ok: false, argv, error: `the console printed no JSON: ${stdoutText.trim() || stderrText.trim()}` }
  }
}

/**
 * argparse's wording for an option a parser does not declare.
 *
 * Matched on the message rather than on the exit code alone, because exit 2 is
 * every console refusal — an unreadable run directory included — and only this
 * one is answered by dropping the flag.
 */
const UNRECOGNIZED = /unrecognized arguments/

/**
 * The same options without the named ones, so an older console is asked the
 * question it understands.
 * @param options - the invocation's options.
 * @param dropped - long option names to leave out.
 * @returns the options that remain.
 */
function withoutActivityOptions(
  options: Readonly<Record<string, string | number | boolean | readonly string[] | undefined>>,
  dropped: readonly string[],
): Readonly<Record<string, string | number | boolean | readonly string[] | undefined>> {
  return Object.fromEntries(Object.entries(options).filter(([name]) => !dropped.includes(name)))
}

/** A record with string keys, as JSON objects decode. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

type Decoded = { kind: 'json'; value: unknown } | { kind: 'not-json' }

/** Parse stdout, tolerating surrounding whitespace and nothing else. */
function decode(text: string): Decoded {
  try {
    return { kind: 'json', value: JSON.parse(text) }
  } catch {
    return { kind: 'not-json' }
  }
}
