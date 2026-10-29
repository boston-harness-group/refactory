/**
 * The interactive graph runner: boots one CEO Agent over the spine, drives it
 * to quiescence (the CEO walks `wf_next → wf_done/wf_verdict`, spawning
 * subagents for content nodes and letting the spine auto-run deterministic
 * nodes), flushes its Session, and reports the best score plus the session log.
 * @module @deepseek-ai/dsh-graph
 */

import { randomUUID } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, renameSync } from 'node:fs'
import { join } from 'node:path'
import { createInterface } from 'node:readline'
import type { Context } from '@deepseek-ai/cordis'
import { brandString } from '@deepseek-ai/dsh-brand'
import { installModelSelection } from '@deepseek-ai/dsh-agent'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { SessionSeq } from '@deepseek-ai/dsh-session'
import type { Session, SessionId, SessionLogOffset } from '@deepseek-ai/dsh-session'
import { CONFIGURED_RUN_SERVICE } from '@deepseek-ai/dsh-workflow-spine'
import type { ConfiguredRun } from '@deepseek-ai/dsh-workflow-spine'
import type {} from '@deepseek-ai/cordis-plugin-loader'
import { GRAPH_STARTUP_SERVICE, STATE_FILES } from './startup.ts'
import type { GraphStartupValues } from './startup.ts'

/** Stable Cordis plugin name. */
export const name = 'graph-runner'

/** Core services required before the CEO turn can start. */
export const inject = ['agents', 'sessions', GRAPH_STARTUP_SERVICE, CONFIGURED_RUN_SERVICE]

/**
 * The best score the mode's ops recorded, or `null` when none recorded one.
 *
 * `null` rather than `0`: an authoring mode has no `score()`, and a run whose state
 * file was never written measured nothing. Reporting either as `0` says "this scored
 * zero", which is a different fact from "this was not measured" and the one a reader
 * would act on.
 */
function bestScore(workdir: string, mode: string): number | null {
  const name = STATE_FILES[mode]
  const candidates = name ? [join(workdir, name)] : []
  for (const file of readdirSync(workdir)) {
    if (file.endsWith('.json')) candidates.push(join(workdir, file))
  }
  for (const path of candidates) {
    if (!existsSync(path)) continue
    try {
      const data = JSON.parse(readFileSync(path, 'utf8')) as { best_score?: unknown }
      if (typeof data.best_score === 'number') return data.best_score
    } catch { /* not JSON yet */ }
  }
  return null
}

/** What the spine last recorded about the cursor, or `undefined` when it recorded nothing. */
function spineStatus(workdir: string): { status: string; nodes?: string[]; completed?: string[]; error?: string } | undefined {
  const path = join(workdir, '.refactory', 'spine-status.json')
  if (!existsSync(path)) return undefined
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as { status: string; nodes?: string[]; completed?: string[]; error?: string }
  } catch {
    // A status file that is not JSON is a status this runner cannot read; the run's
    // own report stands rather than being replaced by a parse error.
    return undefined
  }
}

/** Read worker receipts to get the token total for a completed step. */
function stepTokens(workdir: string, nodeId: string): number {
  const dir = join(workdir, '.refactory', 'workers')
  if (!existsSync(dir)) return 0
  let total = 0
  for (const name of readdirSync(dir)) {
    if (!name.endsWith('.json')) continue
    try {
      const receipt = JSON.parse(readFileSync(join(dir, name), 'utf8'))
      if (receipt?.node_id !== nodeId) continue
      const tokens = receipt?.usage?.tokens ?? receipt?.usage?.known_tokens ?? 0
      if (typeof tokens === 'number' && tokens > 0) total += tokens
    } catch { /* skip unreadable */ }
  }
  return total
}

/** Print done/started lines for step transitions not yet announced. */
function flushStepTransitions(
  workdir: string,
  lastCompleted: string[],
  lastNodes: string[],
  io: { stderr: { write(chunk: string): unknown } },
): { completed: string[]; nodes: string[] } {
  const status = spineStatus(workdir)
  if (status === undefined) return { completed: lastCompleted, nodes: lastNodes }
  const completed = status.completed ?? []
  for (const node of completed) {
    if (!lastCompleted.includes(node)) {
      const tokens = stepTokens(workdir, node)
      io.stderr.write(tokens > 0
        ? `⟩ ${node} done (${tokens.toLocaleString()} tokens)\n`
        : `⟩ ${node} done\n`)
    }
  }
  const nodes = status.nodes ?? []
  for (const node of nodes) {
    if (!lastNodes.includes(node)) io.stderr.write(`⟩ ${node} started\n`)
  }
  return { completed, nodes }
}

/** Flush new assistant messages from the session log and return the new offset. */
function flushMessages(
  agent: Agent,
  lastSeq: SessionLogOffset,
  io: { stdout: { write(chunk: string): unknown } },
): SessionLogOffset {
  for (let seq = lastSeq; seq < agent.session.seq; seq++) {
    const event = agent.session.eventAt(SessionSeq(seq))
    if (event?.type === 'assistant/message') {
      const text = event.data.message.content
        .filter(block => block.type === 'text')
        .map(block => block.text)
        .join('')
      if (text !== '') io.stdout.write(`${text}\n`)
    }
  }
  return agent.session.seq
}

/**
 * Print step transitions and new CEO messages during a turn, by polling the
 * spine-status file and the session log on a short interval. Returns when the
 * agent reaches quiescence.
 *
 * This is what makes "print progress during a turn, not after it" work: one CEO
 * turn can drive several nodes for many minutes, and printing transitions only
 * after `agent.whenIdle()` would leave the terminal silent the whole time.
 */
async function whenIdleWithProgress(
  agent: Agent,
  startup: GraphStartupValues,
  io: Io,
  seqBefore: SessionLogOffset,
): Promise<void> {
  let lastCompleted: string[] = []
  let lastNodes: string[] = []
  let lastSeq = seqBefore
  const poll = setInterval(() => {
    const result = flushStepTransitions(startup.workdir, lastCompleted, lastNodes, io)
    lastCompleted = result.completed
    lastNodes = result.nodes
    // Print new assistant messages as they appear.
    lastSeq = flushMessages(agent, lastSeq, io)
  }, 500)
  try {
    await agent.whenIdle()
  } finally {
    clearInterval(poll)
  }
  // Flush step transitions and messages the poll may not have reached.
  flushStepTransitions(startup.workdir, lastCompleted, lastNodes, io)
  flushMessages(agent, lastSeq, io)
}

/** The final assistant text of the turn. */
function finalText(session: Session, firstSeq: SessionLogOffset): string {
  let text = ''
  for (let seq = firstSeq; seq < session.seq; seq++) {
    const event = session.eventAt(SessionSeq(seq))
    if (event?.type === 'assistant/message') {
      const joined = event.data.message.content
        .filter(block => block.type === 'text')
        .map(block => block.text)
        .join('')
      if (joined !== '') text = joined
    }
  }
  return text
}

/** The latest turn's actual failure; a later successful retry clears an earlier one. */
function turnError(session: Session, firstSeq: SessionLogOffset): string | undefined {
  for (let seq = session.seq - 1; seq >= firstSeq; seq--) {
    const event = session.eventAt(SessionSeq(seq))
    if (event?.type !== 'turn/end') continue
    const reason = event.data.reason
    if (reason.kind === 'completed') return undefined
    if (reason.kind === 'error') return `${reason.error.code}: ${reason.error.message}`
    return `the model turn ended (${reason.kind})`
  }
  return 'the model turn recorded no completion'
}

/** What a run with no model says; the same words as refactory's own refusal. */
export const NO_MODEL_CHOSEN =
  'No model chosen: pick one in Settings → Models or with `refactory model set --provider <provider> --model <model>`.'

/** The terminal the run writes to and reads from; the launcher supplies the process's. */
export interface Io {
  stdout: { write(chunk: string): unknown }
  stderr: { write(chunk: string): unknown }
  /** The person's terminal. A gate reads it too; see {@link chat}. */
  stdin: NodeJS.ReadableStream & { readableEnded?: boolean }
  exit(code: number): void
}

/** One line the person typed, or `undefined` at end of input. */
function readLine(stdin: Io['stdin'], out: Io['stdout']): Promise<string | undefined> {
  // End of input is remembered by the stream itself: a second read after the first
  // exhausted it must not wait for a line that can never come. (A `readline` started
  // on an already-ended stream does not always see the `end` that already fired.)
  if (stdin.readableEnded === true) return Promise.resolve(undefined)
  if (process.stdin.isTTY) out.write('> ')
  const rl = createInterface({ input: stdin, output: process.stdin.isTTY ? process.stdout : undefined })
  return new Promise((resolve) => {
    let settled = false
    const done = (line: string | undefined): void => {
      if (settled) return
      settled = true
      rl.close()
      resolve(line)
    }
    rl.once('close', () => done(undefined))
    rl.question('', line => done(line))
  })
}

/**
 * The live session: print each turn, then let the person say something back.
 *
 * The first turn is the driving task the runner already sent. Progress and
 * assistant messages are printed as they happen during each turn (by polling
 * spine-status.json), not only between turns.
 *
 * It reads the terminal only while the agent is idle, and closes its reader before
 * handing the turn back, so a `user` gate's own prompt owns stdin while the answerer
 * asks — the two never read at the same time.
 */
async function chat(agent: Agent, startup: GraphStartupValues, io: Io, firstSeq: SessionLogOffset): Promise<void> {
  let seqBefore = firstSeq
  for (;;) {
    // Messages were already printed incrementally by whenIdleWithProgress.
    const failure = turnError(agent.session, seqBefore)
    if (failure !== undefined) io.stderr.write(`dsh: ${failure}\n`)
    // Quiescence is not completion: the cursor's own status on disk is what says the
    // graph finished, and there is nothing left to steer once it has.
    if (failure === undefined && spineStatus(startup.workdir)?.status === 'done') return
    const line = await readLine(io.stdin, io.stdout)
    if (line === undefined) return
    const typed = line.trim()
    if (typed === '' || typed === '/quit') return
    seqBefore = agent.session.seq
    agent.followup(createUserMessage({
      content: [{ type: 'text', text: typed }],
      source: { kind: 'user' },
    }))
    await whenIdleWithProgress(agent, startup, io, seqBefore)
  }
}

async function run(ctx: Context, io: Io): Promise<void> {
  const startup = ctx.get(GRAPH_STARTUP_SERVICE) as GraphStartupValues | undefined
  await ctx.get('loader')?.await()
  const configured = ctx.get(CONFIGURED_RUN_SERVICE) as ConfiguredRun | undefined
  if (startup !== undefined && !startup.chat && configured?.automatic === true) {
    // Keep the shipping spine, command/data bridges and status. A graph with no
    // model-owned work needs no CEO just to request its deterministic drain.
    // This path does not interpret prose steering or invent a model session.
    let failure: string | undefined
    try {
      const priorStatus = join(startup.workdir, '.refactory', 'spine-status.json')
      if (existsSync(priorStatus)) {
        renameSync(priorStatus, join(startup.workdir, '.refactory', `spine-status.previous-automatic-${randomUUID()}.json`))
      }
      const result = await configured.advance()
      // A retryable node failure leaves persisted cursor status `ready`. The
      // dispatch result carries the reason; do not replace it with that status.
      if (result !== null && typeof result === 'object' && !Array.isArray(result)
        && result.status === 'failed' && typeof result.error === 'string') {
        failure = result.error
      }
    } catch (error: unknown) {
      failure = error instanceof Error ? error.message : String(error)
    }
    const status = spineStatus(startup.workdir)
    if (failure === undefined && status?.status !== 'done') {
      failure = status === undefined
        ? 'the graph did not start: no readable workflow status was recorded'
        : `the graph did not finish (${status.status})${status.error === undefined ? '' : ` — ${status.error}`}`
    }
    io.stdout.write(JSON.stringify({
      ok: failure === undefined,
      status: status?.status ?? 'not-started',
      ...(failure === undefined ? {} : { error: failure }),
      mode: startup.mode,
      task: startup.taskName,
      budget: startup.budget,
      best_score: bestScore(startup.workdir, startup.mode),
      workdir: startup.workdir,
      driver: 'automatic',
      driver_model_calls: 0,
    }, null, 2) + '\n')
    if (failure !== undefined) io.stderr.write(`dsh: ${failure}\n`)
    io.exit(failure === undefined ? 0 : 1)
    return
  }
  const agents = ctx.get('agents')
  const sessions = ctx.get('sessions')
  if (startup === undefined || agents === undefined || sessions === undefined) return

  // The CEO runs on the route the run names, never on the graph profile's own
  // default: that default is how a terminal run reached a provider nobody chose
  // (dm-258). Refactory always passes both; a launch that passes neither is refused
  // here, before any agent exists, with the message every refusal uses.
  const provider = startup.provider ?? ''
  const model = startup.model ?? ''
  if (provider === '' || model === '') {
    io.stderr.write(`dsh: ${NO_MODEL_CHOSEN}\n`)
    io.exit(1)
    return
  }
  const { agent } = await agents.create({
    sessionId: brandString<SessionId>(`session-${randomUUID()}`),
    meta: { cwd: startup.workdir },
    agentOptions: { provider, model },
    setup: (agentCtx) => {
      // The selection the middleware rewrites each request from has to be the
      // effective one: installing the settings default here pinned every run to it
      // and discarded `--provider`/`--model` without saying so.
      installModelSelection(agentCtx, { current: { provider, model }, assembled: undefined })
    },
  })
  await agent.whenIdle()
  const firstSeq = agent.session.seq
  let caughtError: string | undefined
  try {
    // A reused directory may carry another invocation's done marker. Keep it as
    // history, but require this CEO's workflow calls to write fresh status.
    const priorStatus = join(startup.workdir, '.refactory', 'spine-status.json')
    if (existsSync(priorStatus)) {
      renameSync(priorStatus, join(startup.workdir, '.refactory', `spine-status.previous-${agent.session.id}.json`))
    }
    agent.followup(createUserMessage({
      content: [{ type: 'text', text: startup.task }],
      source: { kind: 'user' },
    }))
    if (startup.chat) {
      await whenIdleWithProgress(agent, startup, io, firstSeq)
      await chat(agent, startup, io, firstSeq)
    } else {
      await agent.whenIdle()
    }
  } catch (error: unknown) {
    caughtError = error instanceof Error ? error.message : String(error)
  }
  await sessions.flush(agent.session)

  const status = spineStatus(startup.workdir)
  const due = status?.nodes === undefined || status.nodes.length === 0
    ? ''
    : `: ${status.nodes.join(', ')} still due`
  const graphError = status === undefined
    ? 'the graph did not start: no readable workflow status was recorded'
    : status.status === 'done'
      ? undefined
      : `the graph did not finish (${status.status}${due})${status.error === undefined ? '' : ` — ${status.error}`}`
  const failure = caughtError ?? turnError(agent.session, firstSeq) ?? graphError
  const result = {
    ok: failure === undefined,
    status: status?.status ?? 'not-started',
    ...(failure === undefined ? {} : { error: failure }),
    mode: startup.mode,
    task: startup.taskName,
    budget: startup.budget,
    best_score: bestScore(startup.workdir, startup.mode),
    session_id: agent.session.id,
    workdir: startup.workdir,
    final: finalText(agent.session, firstSeq).slice(0, 400),
  }
  io.stdout.write(JSON.stringify(result, null, 2) + '\n')

  // A failed final turn (even beside an old done file), absent status, or an
  // unfinished cursor cannot certify completion. Live chat may retry before here.
  if (failure !== undefined) io.stderr.write(`dsh: ${failure}\n`)
  io.exit(failure === undefined ? 0 : 1)
}

/**
 * Mount the interactive runner. The spine row (in the patch) hands this app the
 * same graph + env it runs the CEO against, through {@link GRAPH_STARTUP_SERVICE}.
 * @param ctx - plugin context carrying core services and the launcher exit request.
 * @param io - the terminal, for tests; the process's own streams when omitted.
 */
export function apply(ctx: Context, io?: Io): void {
  const exit = ctx.get('appExit')
  if (exit === undefined) {
    throw new Error('graph-runner: the launcher must provide ctx.appExit before the tree mounts')
  }
  const streams: Io = io ?? {
    stdout: process.stdout,
    stderr: process.stderr,
    stdin: process.stdin,
    exit,
  }
  void run(ctx, streams).catch((error: unknown) => {
    streams.stderr.write(`dsh: ${error instanceof Error ? error.message : String(error)}\n`)
    streams.exit(1)
  })
}
