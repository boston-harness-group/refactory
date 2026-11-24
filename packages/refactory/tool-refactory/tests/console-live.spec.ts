/**
 * The tool against the real loop: a real console process, a real run, real
 * artifacts.
 *
 * The unit spec covers the argv the tool builds; this covers what happens when
 * that argv is executed — which is the only way to know the two halves of the
 * boundary still agree. It runs the loop the way a person pressing "start" in
 * the Web view does, and then reads the run back the way the browser card does.
 */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import * as tool from '../src/index.ts'

/** The checkout that holds `refactory/` and `.venv/`. */
const REPO = join(import.meta.dirname, '..', '..', '..', '..')
/** One run root for the whole file, removed when it ends. */
const ROOT = mkdtempSync(join(tmpdir(), 'refactory-tool-'))

afterAll(() => {
  // A run started here is a separate process that keeps writing into ROOT, so
  // removing the directory can lose a race with it. Retrying is the fix that
  // belongs here; the alternative is a test that fails on timing.
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      rmSync(ROOT, { recursive: true, force: true })
      return
    } catch {
      // Still being written to. Wait and try again.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100)
    }
  }
})

async function setup() {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(LocalSubprocessRuntime)
  await ctx.plugin(tool, {
    toolName: 'refactory',
    repoRoot: REPO,
    pythonBin: join(REPO, '.venv', 'bin'),
    timeoutMs: 120_000,
    collectBytes: 8 * 1024 * 1024,
  })
  return ctx
}

let counter = 0
async function call(ctx: Context, args: Record<string, unknown>) {
  return await ctx.tools.execute({
    signal: new AbortController().signal,
    callId: ToolCallId(`call-${++counter}`),
    name: 'refactory',
    arguments: args,
  })
}

/** The canonical payload the registry validated, as the browser card receives it. */
function payload(result: { meta?: unknown }): { action: string; data: unknown } {
  return result.meta as { action: string; data: unknown }
}

function text(result: { content: { type: string; text?: string }[] }): string {
  return result.content.filter(block => block.type === 'text').map(block => block.text).join('\n')
}

describe('the refactory tool against the real loop', () => {
  it('lists the modes the checkout can actually run', async () => {
    const ctx = await setup()
    const modes = payload(await call(ctx, { action: 'modes' })).data as { id: string; scored: boolean }[]
    const ids = modes.map(mode => mode.id)
    expect(ids).toContain('circle_packing/adaptive')
    expect(modes.find(mode => mode.id === 'circle_packing/adaptive')?.scored).toBe(true)
  }, 120_000)

  it('starts a run, watches it to completion, and reads what it produced', async () => {
    const ctx = await setup()
    const started = payload(await call(ctx, {
      action: 'start',
      mode: 'circle_packing/adaptive',
      runId: 'live',
      runsRoot: ROOT,
      settings: { epochs: 1, batch_size: 2, optimizer: 'random', seed: 3 },
    })).data as { run_id: string; root: string }
    expect(started.run_id).toBe('live')
    expect(readFileSync(join(started.root, 'console.json'), 'utf8')).toContain('circle_packing/adaptive')

    // The run is a separate process, so the call returned before it finished;
    // the loop below is exactly what a reader watching the view does.
    let status: {
      state: string
      finished: boolean
      best_score: number | null
      counters: { evaluations: number }
    }
    const deadline = Date.now() + 120_000
    do {
      status = payload(await call(ctx, { action: 'status', run: started.root })).data as typeof status
      if (!status.finished) await new Promise(resolve => setTimeout(resolve, 200))
    } while (!status.finished && Date.now() < deadline)

    expect(status.state).toBe('finished')
    expect(status.best_score).toBeGreaterThan(0)

    const board = payload(await call(ctx, { action: 'leaderboard', run: started.root })).data as {
      id: string
      score: number
      parent: string | null
      diff: unknown
    }[]
    expect(board.length).toBe(2)
    const child = board.find(row => row.parent !== null)
    expect(child, 'a run must produce at least one candidate with a parent').toBeDefined()

    const one = payload(await call(ctx, {
      action: 'candidate', run: started.root, candidate: child!.id,
    })).data as { parent: string; rendered: string }
    expect(one.parent).toBe(child!.parent)
    expect(one.rendered.length).toBeGreaterThan(0)

    const manifest = payload(await call(ctx, {
      action: 'artifacts', run: started.root, contents: true,
    })).data as { artifacts: { name: string; exists: boolean; content?: unknown }[] }
    const report = manifest.artifacts.find(entry => entry.name === 'report')
    expect(report?.exists).toBe(true)
    expect(String(report?.content)).toContain('#')

    const listed = payload(await call(ctx, { action: 'runs', runsRoot: ROOT })).data as { run_id: string }[]
    expect(listed.map(entry => entry.run_id)).toContain('live')
  }, 180_000)

  it('shows a run making progress while it is still going', async () => {
    const ctx = await setup()
    const started = payload(await call(ctx, {
      action: 'start',
      mode: 'circle_packing/adaptive',
      runId: 'watched',
      runsRoot: ROOT,
      settings: { epochs: 6, batch_size: 3, optimizer: 'random', seed: 11 },
    })).data as { root: string }

    // Poll the way the view does: ask for what is new, and stop as soon as a
    // look lands mid-run. A run that has already finished by the first poll
    // would make this claim vacuous, so the wait is bounded and asserted.
    const deadline = Date.now() + 120_000
    let cursor = 0
    let midFlight: { counters: { evaluations: number }; best_score: number | null } | undefined
    while (Date.now() < deadline) {
      const frame = payload(await call(ctx, { action: 'status', run: started.root, after: cursor })).data as {
        finished: boolean
        counters: { evaluations: number }
        best_score: number | null
        cursor: number
      }
      cursor = frame.cursor
      if (!frame.finished && frame.counters.evaluations > 0) { midFlight = frame; break }
      if (frame.finished) break
    }
    expect(midFlight, 'the run finished before any poll could see it in flight').toBeDefined()
    expect(midFlight!.counters.evaluations).toBeGreaterThan(0)

    // Leave nothing running: this test deliberately catches the run mid-flight,
    // so without this the loop would keep writing into a directory the file
    // removes when it ends.
    let stopped = false
    const stopDeadline = Date.now() + 60_000
    while (!stopped && Date.now() < stopDeadline) {
      const stop = payload(await call(ctx, { action: 'stop', run: started.root })).data as { stopped: boolean }
      stopped = stop.stopped
      if (!stopped) await new Promise(resolve => setTimeout(resolve, 200))
    }
    expect(stopped, 'the run would not stop, so it outlives this test').toBe(true)
  }, 180_000)

  it('reports an unknown run as the console\'s own refusal', async () => {
    const ctx = await setup()
    const result = await call(ctx, { action: 'status', run: join(ROOT, 'no-such-run') })
    expect(result.isError).toBe(true)
    expect(text(result)).toContain('does not exist (checked')
  }, 60_000)

  it('summarizes a real status call without dumping the payload', async () => {
    const ctx = await setup()
    const modes = await call(ctx, { action: 'modes' })
    const summary = text(modes)
    expect(summary).toContain('circle_packing/adaptive')
    // The real catalogue's own fields, rendered: both routes, and the gate that
    // belongs to a person (#92, #93). A renderer that dropped them would still
    // list every mode and pass a test that only looked for ids.
    expect(summary).toContain('or wf_open to drive it node by node')
    expect(summary).toContain('gate_strategy (user)')
    expect(summary).toContain('knobs: ')
    // About one line per mode: a dumped payload runs to hundreds; the catalogue grows.
    expect(summary.split('\n').length).toBeLessThan(60)
  }, 120_000)
})
