/**
 * Parallel data nodes on the spine: the `parallelism` field on the views and the
 * pass, and the concurrent drain end to end -- overlap, item order, a `wf_open`
 * run, resume of a half-finished pass, and the recorded fallback.
 */
import { exec, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { describe, expect, it, afterEach } from 'vitest'
import * as spinePlugin from '../src/index.ts'
import type { Config } from '../src/index.ts'
import { Spine, validateGraph } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-spine-data-parallel-'))
  dirs.push(dir)
  return dir
}

/** A data graph with a configurable parallelism field. */
function parallelDataGraph(parallelism?: number): WorkflowGraph {
  return {
    name: 'parallel-data',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data',
        task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'end',
        reads: [], writes: ['current_item.json'],
        ...(parallelism !== undefined ? { parallelism } : {}),
      },
      work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
      end: { _type: 'FnNode', id: 'end', command: 'true', reads: [], writes: [] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'data', target: 'after' },
      { source: 'work', target: 'end' },
    ],
  }
}

describe('data-parallel: spine parallelism field', () => {
  it('T2 — exposes parallelism > 1 on the NodeView', () => {
    const spine = new Spine(parallelDataGraph(4))
    const next = spine.next()
    expect(next.status).toBe('ready')
    const dataView = next.nodes!.find(n => n.id === 'data')
    expect(dataView).toBeDefined()
    expect(dataView!.parallelism).toBe(4)
  })

  it('T2 — omits parallelism when 1 (default)', () => {
    const spine = new Spine(parallelDataGraph())
    const next = spine.next()
    const dataView = next.nodes!.find(n => n.id === 'data')
    expect(dataView).toBeDefined()
    expect(dataView!.parallelism).toBeUndefined()
  })

  it('T2 — startPass stores parallelism in DataPass', () => {
    const spine = new Spine(parallelDataGraph(4))
    spine.startPass('data', 3)
    const pass = spine.passOf()
    expect(pass).toBeDefined()
    expect(pass!.parallelism).toBe(4)
  })

  it('T2 — DataPass has no parallelism when default', () => {
    const spine = new Spine(parallelDataGraph())
    spine.startPass('data', 3)
    const pass = spine.passOf()
    expect(pass).toBeDefined()
    expect(pass!.parallelism).toBeUndefined()
  })

  it('T2 — subgraphFor returns the data body nodes', () => {
    const spine = new Spine(parallelDataGraph(4))
    spine.startPass('data', 3)
    const pass = spine.passOf()!
    const sub = spine.subgraphFor(pass.entry, pass.exit)
    expect(sub.body.has('work')).toBe(true)
    expect(sub.body.has('end')).toBe(true)
    expect(sub.body.has('data')).toBe(false)
    expect(sub.body.has('after')).toBe(false)
    expect(Object.keys(sub.nodes)).toEqual(expect.arrayContaining(['work', 'end']))
    expect(sub.edges.length).toBe(1)
    expect(sub.edges[0]).toMatchObject({ source: 'work', target: 'end' })
  })
})

// ── The concurrent drain, end to end, through the real item loader ──────────
//
// The plugin drives `python -m refactory_dsh.items` (the run's own interpreter,
// as execute.spec does) and runs each body command in its own process, so items
// really overlap and the pass state, `item_scores` and pass directories are the
// loader's own.

const PYTHON = resolve('.venv/bin/python')

/** The Task the loader judges each pass with: an even item passes, an odd one fails. */
const TASK_MODULE = `from pathlib import Path
from refactory_dsh.task import Task, VerifyResult

class PTask(Task):
    name = "ptask"
    def verify(self, instance, workspace):
        out = (Path(workspace) / "out.txt").read_text().strip()
        passed = out == instance.id and int(instance.id[1:]) % 2 == 0
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0, details=out)
`

/**
 * The body command: note when it starts and ends, sleep the item's delay, write the
 * answer. It runs in the item's pass directory, where `begin` put the item.
 */
const WORK = `${PYTHON} -c 'import json,time,pathlib; d=json.load(open("current_item.json")); `
  + `pathlib.Path("start").write_text(repr(time.time())); time.sleep(float(d["delay"])); `
  + `pathlib.Path("end").write_text(repr(time.time())); pathlib.Path("out.txt").write_text(d["id"])'`

/**
 * WORK, except that after noting its start the item waits until `ready` (a Python
 * expression, run in the item's pass directory) holds, then sleeps and ends. A test
 * that measures overlap sets it with this gate instead of hoping the scheduler starts
 * the items close enough together. The wait gives up after `timeoutS` and the item
 * ends anyway, so a drain that never admits enough items fails the test's assertion
 * instead of hanging it.
 */
function gatedWork(ready: string, timeoutS: number): string {
  return `${PYTHON} -c 'import glob,json,os,time,pathlib; d=json.load(open("current_item.json")); `
    + `pathlib.Path("start").write_text(repr(time.time())); t=time.time()+${timeoutS}; `
    + `[time.sleep(0.02) for _ in iter(lambda: not (${ready}) and time.time()<t, False)]; time.sleep(float(d["delay"])); `
    + `pathlib.Path("end").write_text(repr(time.time())); pathlib.Path("out.txt").write_text(d["id"])'`
}

/** The gate for `count` items to have started in this pass, counting their own `start` notes. */
const startedTogether = (count: number): string => `len(glob.glob("../*/start"))>=${count}`

function loaderGraph(parallelism: number, work = WORK): WorkflowGraph {
  return {
    name: 'parallel-loader',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data', task_ref: 'ptask:PTask',
        subgraph_entry: 'work', subgraph_exit: 'work',
        reads: [], writes: ['current_item.json'],
        ...(parallelism > 1 ? { parallelism } : {}),
      },
      work: { _type: 'FnNode', id: 'work', command: work, reads: ['current_item.json'], writes: ['out.txt'] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [{ source: 'data', target: 'after' }],
  }
}

/** A run directory: the Task module and the declarations the lowering would leave. */
function loaderRun(items: Array<Record<string, unknown>>, parallelism: number): { workdir: string; taskDir: string } {
  const workdir = scratch()
  const taskDir = scratch()
  writeFileSync(join(taskDir, 'ptask.py'), TASK_MODULE)
  const script = [
    'import json, sys',
    'from pathlib import Path',
    'from refactory_dsh import nn',
    'from refactory_dsh.lowering import write_declarations',
    'g = nn.Graph()',
    "g.add(nn.Node('data', 'data', subgraph_entry='work', subgraph_exit='work', writes=('current_item.json',),"
      + " produces=('item_scores',), task_ref='ptask:PTask', inline_items=tuple(json.loads(sys.argv[2])),"
      + ' parallelism=int(sys.argv[3])))',
    "g.add(nn.Node('work', 'op', call='ptask:work', reads=('current_item.json',), writes=('out.txt',)))",
    "g.add(nn.Node('after', 'op', call='ptask:after'))",
    "g.connect('data', 'after')",
    "write_declarations({'graph': g.to_state()}, Path(sys.argv[1]))",
  ].join('\n')
  const written = spawnSync(PYTHON, ['-c', script, workdir, JSON.stringify(items), String(parallelism)], { encoding: 'utf8' })
  expect(written.status, written.stderr).toBe(0)
  return { workdir, taskDir }
}

interface Ran { command: string; workdir: string; started: number; ended: number }

/**
 * A shell that runs each command in its own process without blocking the event
 * loop, so concurrent commands overlap. A command `hold` matches never settles: a
 * run stopped part way through a pass.
 */
function processShell(ran: Ran[], hold?: (command: string, workdir: string) => boolean): object {
  return {
    resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string> }) => ({
      command: request.command, workdir: request.workdir ?? process.cwd(),
      timeoutMs: request.timeoutMs ?? 60_000, onExpiry: 'kill', stdoutMaxBytes: 1_000_000, env: request.env,
    }),
    execute: async (spec: { command: string; workdir: string; timeoutMs: number; env?: Record<string, string> }) => ({
      result: async () => {
        if (hold?.(spec.command, spec.workdir) === true) return new Promise(() => {})
        const started = Date.now()
        return new Promise(settle => {
          exec(spec.command, { cwd: spec.workdir, env: { ...process.env, ...spec.env }, shell: '/bin/sh', maxBuffer: 10_000_000 },
            (error, stdout, stderr) => {
              ran.push({ command: spec.command, workdir: spec.workdir, started, ended: Date.now() })
              const code = error === null ? 0 : typeof error.code === 'number' ? error.code : 1
              settle({
                exitCode: code, signal: null, timedOut: false, aborted: false, timeoutMs: spec.timeoutMs,
                stdout: { text: stdout, truncated: false }, stderr: { text: stderr, truncated: false },
              })
            })
        })
      },
    }),
  }
}

async function mountLoader(shell: object, overrides: Partial<Config>): Promise<Context> {
  const root = new Context()
  await root.plugin(SystemPrompt)
  await root.plugin(ToolRuntime)
  root.provide('shell', shell)
  await root.plugin(spinePlugin, {
    toolPrefix: 'wf', maxIterations: 3, interactiveOpen: false, openCommand: '', openArgs: [],
    executeCommands: true, commandEnv: {}, commandTimeoutMs: 60_000, commandSandboxMode: 'danger-full-access',
    dataCommand: `${PYTHON} -m refactory_dsh.items`, steerCommand: '', steerArgs: [], ceoDeny: [],
    ...overrides,
  } as Config)
  return root
}

async function call(ctx: Context, name: string, args: unknown = {}, extra: unknown = {}): Promise<Record<string, unknown>> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} is not registered`)
  return await definition.execute(args, extra as never) as Record<string, unknown>
}

/** Call `wf_next` until the run is done, as a driver with nothing to do would. */
async function drive(ctx: Context, extra: unknown = {}): Promise<Record<string, unknown>> {
  for (let turn = 0; turn < 20; turn += 1) {
    const state = await call(ctx, 'wf_next', {}, extra)
    if (state.status !== 'ready') return state
  }
  throw new Error('the run did not finish in 20 turns')
}

/** The most body commands that were running at one moment, from their own clocks. */
function peakOverlap(workdir: string): number {
  const passes = join(workdir, 'passes', 'data')
  const spans = readdirSync(passes).map(dir => ({
    start: Number(readFileSync(join(passes, dir, 'start'), 'utf8')),
    end: Number(readFileSync(join(passes, dir, 'end'), 'utf8')),
  }))
  return Math.max(...spans.map(span => spans.filter(other => other.start <= span.start && span.start < other.end).length))
}

function facts(workdir: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(workdir, 'facts.json'), 'utf8')) as Record<string, unknown>
}

/** Eight items; the first sleeps longest, so the items end in reverse order. */
const ITEMS = Array.from({ length: 8 }, (_, index) => ({ id: `i${index}`, delay: (8 - index) * 0.15 }))

describe('data-parallel: the concurrent drain', () => {
  it('runs items four at a time, keeping item_scores and records in item order', async () => {
    const results: Record<number, { wall: number; peak: number; scores: unknown; records: unknown; begins: string[] }> = {}
    for (const parallelism of [1, 4]) {
      const { workdir, taskDir } = loaderRun(ITEMS, parallelism)
      const ran: Ran[] = []
      // Each item waits for the first `parallelism` items to start, so the peak is
      // the drain's bound, not the luck of the scheduler (a no-op at parallelism 1).
      const ctx = await mountLoader(processShell(ran), {
        graph: loaderGraph(parallelism, gatedWork(startedTogether(parallelism), 10)), workdir,
        commandEnv: { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir },
      })
      const started = Date.now()
      expect(await drive(ctx)).toMatchObject({ status: 'done' })
      const scores = facts(workdir).item_scores as Array<Record<string, unknown>>
      results[parallelism] = {
        wall: Date.now() - started,
        peak: peakOverlap(workdir),
        scores: scores.map(({ id, passed, score }) => ({ id, passed, score })),
        records: JSON.parse(readFileSync(join(workdir, 'item_records.json'), 'utf8')),
        begins: ran.filter(entry => entry.command.includes(' begin ')).map(entry => entry.command),
      }
      if (parallelism > 1) {
        // The node is reset once, before any item begins; no begin resets it.
        const prepare = ran.findIndex(entry => entry.command.includes(' prepare '))
        const firstBegin = ran.findIndex(entry => entry.command.includes(' begin '))
        expect(prepare).toBeGreaterThanOrEqual(0)
        expect(ran[prepare]!.ended).toBeLessThanOrEqual(ran[firstBegin]!.started)
        expect(results[parallelism]!.begins.every(command => command.includes("'--concurrent'"))).toBe(true)
        expect(ran.some(entry => entry.command.includes(' finish '))).toBe(true)
      }
      console.log(`spine parallelism=${parallelism}: wall ${results[parallelism]!.wall} ms, peak ${results[parallelism]!.peak} items at once`)
    }
    expect(results[1]!.peak).toBe(1)
    expect(results[4]!.peak).toBe(4)
    expect(results[4]!.wall).toBeLessThan(results[1]!.wall)
    // Item order, though item 7 ended first.
    expect((results[4]!.scores as Array<{ id: string }>).map(entry => entry.id)).toEqual(ITEMS.map(item => item.id))
    expect(results[4]!.scores).toEqual(results[1]!.scores)
    expect(results[4]!.records).toEqual(results[1]!.records)
  }, 60_000)

  it('runs a wf_open run concurrently, with no configured graph', async () => {
    const { workdir, taskDir } = loaderRun(ITEMS.slice(0, 4), 4)
    const graphPath = join(workdir, 'graphs', 'parallel.graph.json')
    mkdirSync(join(workdir, 'graphs'), { recursive: true })
    writeFileSync(graphPath, JSON.stringify(loaderGraph(4, gatedWork(startedTogether(4), 10))))
    const stub = join(scratch(), 'open-stub')
    writeFileSync(stub, `#!/bin/sh\nprintf '%s' '${JSON.stringify({
      mode: 'pkg/parallel', run: workdir, name: 'parallel', graph: graphPath,
      declarations: join(workdir, '.refactory', 'declarations.json'),
      commandEnv: { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir },
      dataCommand: `${PYTHON} -m refactory_dsh.items`,
    })}'\n`)
    chmodSync(stub, 0o755)
    const ran: Ran[] = []
    const ctx = await mountLoader(processShell(ran), { interactiveOpen: true, openCommand: stub })
    const session = { agent: { session: { id: 'session-1', requestHeader: () => undefined } } }
    await call(ctx, 'wf_open', { mode: 'pkg/parallel' }, session)
    expect(await drive(ctx, session)).toMatchObject({ status: 'done' })
    expect(peakOverlap(workdir)).toBe(4)
    expect((facts(workdir).item_scores as Array<{ id: string }>).map(entry => entry.id)).toEqual(['i0', 'i1', 'i2', 'i3'])
  }, 60_000)

  it('resumes a half-finished concurrent pass from the items that finished', async () => {
    const { workdir, taskDir } = loaderRun(ITEMS.map(item => ({ ...item, delay: 0.05 })), 2)
    const env = { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir }
    // The first run stops with items 0 and 1 finished and items 2 and 3 mid-walk.
    const stopped = (command: string, dir: string): boolean =>
      command === WORK && !/_[01]$/.test(dir)
    const first = await mountLoader(processShell([], stopped), { graph: loaderGraph(2), workdir, commandEnv: env })
    void call(first, 'wf_next')
    const passes = join(workdir, 'passes', 'data')
    for (let wait = 0; wait < 200; wait += 1) {
      const dirs = existsSync(passes) ? readdirSync(passes) : []
      if (dirs.filter(dir => existsSync(join(passes, dir, '.done'))).length === 2 && dirs.length === 4) break
      await new Promise(settle => setTimeout(settle, 50))
    }
    const finishedBefore = readdirSync(passes).filter(dir => existsSync(join(passes, dir, '.done'))).sort()
    expect(finishedBefore).toEqual(['i0_0', 'i1_1'])
    const doneAt = statSync(join(passes, 'i0_0', '.done')).mtimeMs

    const saved = readFileSync(join(workdir, '.refactory/spine-checkpoint.json'))
    expect(JSON.parse(saved.toString()).state?.pass).toMatchObject({ node: 'data', parallelism: 2 })
    // The run died mid-pass (status `executing`): a person acknowledges that before
    // the resume, as for any node whose side effects are unknown.
    expect(JSON.parse(saved.toString()).status).toBe('executing')
    writeFileSync(join(workdir, '.refactory/resume-request.json'), JSON.stringify({
      checkpoint_sha256: createHash('sha256').update(saved).digest('hex'),
      acknowledge_executing: true,
    }))
    const ran: Ran[] = []
    const resumed = await mountLoader(processShell(ran), { graph: loaderGraph(2), workdir, commandEnv: env })
    expect(await drive(resumed)).toMatchObject({ status: 'done' })

    // Items 0 and 1 were not walked again, and their passes were kept, not deleted.
    const walked = ran.filter(entry => entry.command === WORK).map(entry => entry.workdir.split('/').pop()).sort()
    expect(walked).toEqual(['i2_2', 'i3_3', 'i4_4', 'i5_5', 'i6_6', 'i7_7'])
    expect(ran.some(entry => entry.command.includes(' prepare '))).toBe(false)
    expect(statSync(join(passes, 'i0_0', '.done')).mtimeMs).toBe(doneAt)
    expect((facts(workdir).item_scores as Array<{ id: string }>).map(entry => entry.id)).toEqual(ITEMS.map(item => item.id))
    expect((JSON.parse(readFileSync(join(workdir, 'item_records.json'), 'utf8')) as Array<{ id: string }>).map(entry => entry.id))
      .toEqual(ITEMS.map(item => item.id))
  }, 60_000)

  /** Stop the first run inside `prepare`, then resume it, acknowledging the stop. */
  async function stopInPrepareThenResume(graph: WorkflowGraph, workdir: string, env: Record<string, string>, ran: Ran[], once = false): Promise<Record<string, unknown>> {
    let held = false
    const first = await mountLoader(processShell([], command => {
      if (!command.includes(' prepare ')) return false
      held = true
      return true
    }), { graph, workdir, commandEnv: env })
    void call(first, 'wf_next')
    for (let wait = 0; wait < 200 && !held; wait += 1) await new Promise(settle => setTimeout(settle, 25))
    expect(held).toBe(true)
    const saved = readFileSync(join(workdir, '.refactory/spine-checkpoint.json'))
    // The pass is not checkpointed as open until its node is prepared for this visit.
    expect(JSON.parse(saved.toString()).state?.pass).toBeUndefined()
    writeFileSync(join(workdir, '.refactory/resume-request.json'), JSON.stringify({
      checkpoint_sha256: createHash('sha256').update(saved).digest('hex'),
      acknowledge_executing: true,
    }))
    const resumed = await mountLoader(processShell(ran), { graph, workdir, commandEnv: env })
    return once ? await call(resumed, 'wf_next') : await drive(resumed)
  }

  it('never takes an earlier visit\'s finished passes for this visit\'s, across a stop', async () => {
    const { workdir, taskDir } = loaderRun(ITEMS.slice(0, 4).map(item => ({ ...item, delay: 0.05 })), 4)
    // An earlier visit of the node finished item 0, with a different answer.
    const stale = join(workdir, 'passes', 'data', 'i0_0')
    mkdirSync(stale, { recursive: true })
    writeFileSync(join(stale, '.item_record.json'), JSON.stringify({ id: 'i0', passed: false, score: 0, stale: true }))
    writeFileSync(join(stale, '.done'), '')
    const ran: Ran[] = []
    const env = { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir }
    expect(await stopInPrepareThenResume(loaderGraph(4), workdir, env, ran)).toMatchObject({ status: 'done' })
    const walked = ran.filter(entry => entry.command === WORK).map(entry => entry.workdir.split('/').pop()).sort()
    expect(walked).toEqual(['i0_0', 'i1_1', 'i2_2', 'i3_3'])
    const records = JSON.parse(readFileSync(join(workdir, 'item_records.json'), 'utf8')) as Array<Record<string, unknown>>
    expect(records.some(record => 'stale' in record)).toBe(false)
    expect(records[0]).toMatchObject({ id: 'i0', passed: true })
  }, 60_000)

  it('records the fallback on a run resumed before its pass opened', async () => {
    const graph = loaderGraph(4)
    graph.nodes.work = { _type: 'AgentNode', id: 'work', role: 'builder', prompt_template: 'W', reads: [], writes: [] }
    const { workdir, taskDir } = loaderRun(ITEMS.slice(0, 2), 4)
    expect(await stopInPrepareThenResume(graph, workdir, { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir }, [], true))
      .toMatchObject({ status: 'ready', nodes: [{ id: 'work' }] })
    expect(facts(workdir).data_parallelism_effective).toMatchObject({ node: 'data', requested: 4, effective: 1 })
  }, 60_000)

  it('records why a body the driving agent must run keeps its items one at a time', async () => {
    const graph = loaderGraph(4)
    graph.nodes.work = { _type: 'AgentNode', id: 'work', role: 'builder', prompt_template: 'W', reads: [], writes: [] }
    const { workdir, taskDir } = loaderRun(ITEMS.slice(0, 2), 4)
    const ran: Ran[] = []
    const ctx = await mountLoader(processShell(ran), { graph, workdir, commandEnv: { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir } })
    const state = await call(ctx, 'wf_next')
    // The sequential drain hands the agent node over, one item at a time.
    expect(state).toMatchObject({ status: 'ready', nodes: [{ id: 'work' }] })
    expect(facts(workdir).data_parallelism_effective).toEqual({
      node: 'data', requested: 4, effective: 1,
      reason: "node 'work' (AgentNode) has no command, so the driving agent runs it",
    })
    const events = readFileSync(join(workdir, '.refactory', 'events.jsonl'), 'utf8')
    expect(events).toContain('data parallelism fallback to serial: node')
    expect(ran.filter(entry => entry.command.includes(' begin ')).every(entry => !entry.command.includes('--concurrent'))).toBe(true)
  }, 60_000)
})

/**
 * The body command for a failed item: item `id` prints `stderr` and exits 1 (after
 * recording its own step as missing, when `missing`); every other item answers.
 */
function failOn(id: string, stderr: string, missing: boolean): string {
  return `${PYTHON} -c 'import json,sys,pathlib; d=json.load(open("current_item.json")); p=pathlib.Path("reviews"); `
    + `(d["id"] != ${JSON.stringify(id)}) or (${missing ? 'p.mkdir(exist_ok=True), (p / "work-missing.txt").write_text("spent"), ' : ''}`
    + `sys.stderr.write(${JSON.stringify(stderr)}), sys.exit(1)); pathlib.Path("out.txt").write_text(d["id"])'`
}

describe('data-parallel: a failed body node in its item record', () => {
  const THREE = ITEMS.slice(0, 3)

  async function failedRecord(parallelism: number, work: string): Promise<Record<string, unknown>> {
    const { workdir, taskDir } = loaderRun(THREE, parallelism)
    const ctx = await mountLoader(processShell([]), {
      graph: loaderGraph(parallelism, work), workdir,
      commandEnv: { REFACTORY_WORKDIR: workdir, PYTHONPATH: taskDir },
    })
    expect(await drive(ctx)).toMatchObject({ status: 'done' })
    const records = JSON.parse(readFileSync(join(workdir, 'item_records.json'), 'utf8')) as Array<Record<string, unknown>>
    return records.find(record => record.id === 'i1')!
  }

  it.each([[1], [4]])('names the node and its error, then keeps the stderr tail (parallelism %i)', async (parallelism) => {
    const stderr = 'loading notes.txt\nSnippetError: verdict gate check: no answer\n--- refactory failure ---\nSnippetError: verdict gate check: no answer\n'
    const record = await failedRecord(parallelism, failOn('i1', stderr, false))
    const text = String(record.error_text)
    expect(text.split('\n')[0]).toBe('work: SnippetError: verdict gate check: no answer')
    // The stderr tail, without the runner's copy of its account before the marker.
    expect(text.slice(text.indexOf('\n') + 1).trimEnd())
      .toBe('loading notes.txt\n--- refactory failure ---\nSnippetError: verdict gate check: no answer')
  }, 60_000)

  it.each([[1], [4]])('finds no answer from a file named only before the marker (parallelism %i)', async (parallelism) => {
    // items.no_answer reads the whole text: the step's missing output, out.txt, is named
    // only in a log line the runner printed before its marker.
    const stderr = 'cannot read out.txt\n--- refactory failure ---\nRuntimeError: step failed\n'
    const record = await failedRecord(parallelism, failOn('i1', stderr, true))
    expect(record.error_text).toBeNull()
    expect(String(record.fail_reason)).toMatch(/^no answer: work/)
  }, 60_000)
})

describe('data-parallel: validates graph with parallelism', () => {
  it('accepts a data node with parallelism', () => {
    const issues = validateGraph(parallelDataGraph(4))
    expect(issues.length).toBe(0)
  })
})

describe('data-parallel: non-deterministic body detection', () => {
  it('subgraphFor returns nodes that can be checked for determinism', () => {
    // A data body with an AgentNode (like LiveMath's solve) is NOT fully
    // deterministic, so the drain should fall back to sequential.
    const graphWithAgent: WorkflowGraph = {
      name: 'agent-data',
      start_node: 'data',
      nodes: {
        data: {
          _type: 'DataNode', id: 'data',
          task_ref: 'probe', subgraph_entry: 'prep', subgraph_exit: 'solve',
          reads: [], writes: ['current_item.json'],
          parallelism: 4,
        },
        prep: { _type: 'FnNode', id: 'prep', command: 'echo ok', reads: [], writes: [] },
        solve: { _type: 'AgentNode', id: 'solve', reads: [], writes: ['answer.txt'] },
        after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
      },
      edges: [
        { source: 'data', target: 'after' },
        { source: 'prep', target: 'solve' },
      ],
    }
    const spine = new Spine(graphWithAgent)
    spine.startPass('data', 3)
    const pass = spine.passOf()!
    expect(pass.parallelism).toBe(4)

    // Check the body nodes via the spine's public view() method.
    const sub = spine.subgraphFor(pass.entry, pass.exit)
    const bodyViews = Object.keys(sub.nodes).map(id => spine.view(id))

    // 'prep' is a FnNode with a command → deterministic.
    const prepView = bodyViews.find(v => v.id === 'prep')!
    expect(prepView._type).toBe('FnNode')
    expect(prepView.command).toBe('echo ok')

    // 'solve' is an AgentNode without dispatch_command → NOT deterministic.
    const solveView = bodyViews.find(v => v.id === 'solve')!
    expect(solveView._type).toBe('AgentNode')
    // AgentNode without a dispatch command has no command field set.
    expect(solveView.command).toBeUndefined()

    // The drain should detect this and NOT enter drainConcurrentData.
    // isDeterministic(solveView) would be false.
  })
})

// ── Slot-directory gating (correction 1): shared cross-evaluation item limit ──

describe('data-parallel: slot-directory gating', () => {
  it('two concurrent drains against one slot dir share its N slots: one drain can hold all N while the other waits, never more than N run, and every slot is released', async () => {
    const N = 4
    const itemsPerDrain = 6
    const slotDir = join(scratch(), 'item-slots')
    mkdirSync(slotDir, { recursive: true })
    const items = Array.from({ length: itemsPerDrain }, (_, i) => ({ id: `d${i}`, delay: 0 }))
    // Every item holds its slot until the test opens the gate, so the overlap is set
    // by the test, not by how the scheduler happens to start the drains.
    const gate = join(scratch(), 'open')
    const work = gatedWork(`os.path.exists("${gate}")`, 45)

    /** The items of a drain that have started their body and not ended it. */
    const running = (workdir: string): number => {
      const passes = join(workdir, 'passes', 'data')
      if (!existsSync(passes)) return 0
      return readdirSync(passes).filter(dir => existsSync(join(passes, dir, 'start')) && !existsSync(join(passes, dir, 'end'))).length
    }
    const until = async (what: string, ready: () => boolean): Promise<void> => {
      for (const deadline = Date.now() + 20_000; !ready(); await new Promise(settle => setTimeout(settle, 25))) {
        if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`)
      }
    }

    // Set process.env so drainConcurrentData reads the slot directory
    const origSlotsDir = process.env.REFACTORY_ITEM_SLOTS_DIR
    const origSlots = process.env.REFACTORY_ITEM_SLOTS
    process.env.REFACTORY_ITEM_SLOTS_DIR = slotDir
    process.env.REFACTORY_ITEM_SLOTS = String(N)

    try {
      // Local parallelism above N, so only the shared slots bound the items.
      const drain1Run = loaderRun(items, itemsPerDrain)
      const drain2Run = loaderRun(items, itemsPerDrain)
      const ran1: Ran[] = []
      const ran2: Ran[] = []

      const ctx1 = await mountLoader(processShell(ran1), {
        graph: loaderGraph(itemsPerDrain, work), workdir: drain1Run.workdir,
        commandEnv: { REFACTORY_WORKDIR: drain1Run.workdir, PYTHONPATH: drain1Run.taskDir },
      })
      const ctx2 = await mountLoader(processShell(ran2), {
        graph: loaderGraph(itemsPerDrain, work), workdir: drain2Run.workdir,
        commandEnv: { REFACTORY_WORKDIR: drain2Run.workdir, PYTHONPATH: drain2Run.taskDir },
      })

      let drain1: Promise<Record<string, unknown>> | undefined
      let drain2: Promise<Record<string, unknown>> | undefined
      let settled: Array<PromiseSettledResult<Record<string, unknown> | undefined>>
      try {
        drain1 = drive(ctx1)
        // One drain alone takes every slot: it is not held to a share (N/2) of them.
        await until(`the first drain to hold all ${N} slots`, () => running(drain1Run.workdir) >= N)
        drain2 = drive(ctx2)
        await until('the second drain to prepare its pass', () => ran2.some(entry => entry.command.includes(' prepare ')))
        await new Promise(settle => setTimeout(settle, 500))
        // The N slots are one set for both drains: while the first holds all of them
        // the second starts nothing, and the first runs N items though it may run 6.
        expect(running(drain2Run.workdir)).toBe(0)
        expect(running(drain1Run.workdir)).toBe(N)
        expect(readdirSync(slotDir)).toHaveLength(N)
      } finally {
        writeFileSync(gate, '')
        settled = await Promise.allSettled([drain1, drain2])
      }
      expect(settled).toMatchObject([
        { status: 'fulfilled', value: { status: 'done' } },
        { status: 'fulfilled', value: { status: 'done' } },
      ])

      // Collect all start/end times from both drains
      const allSpans: Array<{ start: number; end: number }> = []
      for (const workdir of [drain1Run.workdir, drain2Run.workdir]) {
        const passes = join(workdir, 'passes', 'data')
        for (const dir of readdirSync(passes)) {
          allSpans.push({
            start: Number(readFileSync(join(passes, dir, 'start'), 'utf8')),
            end: Number(readFileSync(join(passes, dir, 'end'), 'utf8')),
          })
        }
      }
      expect(allSpans).toHaveLength(2 * itemsPerDrain)

      // Compute peak overlap across both drains
      const peak = Math.max(...allSpans.map(span =>
        allSpans.filter(other => other.start <= span.start && span.start < other.end).length
      ))

      console.log(`slot-dir gating: peak=${peak}, N=${N}, spans=${allSpans.length}`)

      // Never exceeded N, and reached it (the gate held N items at once).
      expect(peak).toBe(N)
      // Every slot was released by the drain that held it.
      expect(readdirSync(slotDir)).toEqual([])
    } finally {
      // Restore env
      if (origSlotsDir === undefined) delete process.env.REFACTORY_ITEM_SLOTS_DIR
      else process.env.REFACTORY_ITEM_SLOTS_DIR = origSlotsDir
      if (origSlots === undefined) delete process.env.REFACTORY_ITEM_SLOTS
      else process.env.REFACTORY_ITEM_SLOTS = origSlots

      // Clean up slot files
      for (let k = 0; k < N; k++) {
        const slotPath = join(slotDir, `slot-${k}`)
        if (existsSync(slotPath)) rmSync(slotPath)
      }
    }
  }, 60_000)

  it('a slot whose holder pid is gone is reclaimed through the real claim, and no tombstone is left', () => {
    const slotDir = join(scratch(), 'item-slots')
    mkdirSync(slotDir, { recursive: true })
    const dead = spawnSync('true')
    expect(dead.pid).toBeGreaterThan(0)
    writeFileSync(join(slotDir, 'slot-0'), `${dead.pid}:stale`)
    const token = spinePlugin.itemSlotToken()
    expect(spinePlugin.tryAcquireItemSlot(slotDir, 1, token)).toBe(0)
    expect(readFileSync(join(slotDir, 'slot-0'), 'utf8')).toBe(token)
    expect(readdirSync(slotDir)).toEqual(['slot-0'])
  })

  it('a claimant that loses the reclaim race leaves the winner its slot', () => {
    const slotDir = join(scratch(), 'item-slots')
    mkdirSync(slotDir, { recursive: true })
    const dead = spawnSync('true')
    const slot = join(slotDir, 'slot-0')
    writeFileSync(slot, `${dead.pid}:stale`)
    const winner = `${process.pid}:winner`
    // Between our read of the dead token and our move, another claimant
    // reclaims the slot and holds it.
    const result = spinePlugin.tryAcquireItemSlot(slotDir, 1, spinePlugin.itemSlotToken(), () => {
      rmSync(slot)
      writeFileSync(slot, winner)
    })
    expect(result).toBe(-1)
    expect(readFileSync(slot, 'utf8')).toBe(winner)
    expect(readdirSync(slotDir)).toEqual(['slot-0'])
  })

  it('the spine source loads node:fs only through its imports (an ESM build has no require)', () => {
    const source = readFileSync(resolve('packages/workflow/workflow-spine/src/index.ts'), 'utf8')
    expect(source).not.toMatch(/\brequire\(/)
  })

  it('a slot whose holder is alive, or still writing its token, is not reclaimed', () => {
    const slotDir = join(scratch(), 'item-slots')
    mkdirSync(slotDir, { recursive: true })
    writeFileSync(join(slotDir, 'slot-0'), `${process.pid}:other`)
    writeFileSync(join(slotDir, 'slot-1'), '')
    expect(spinePlugin.tryAcquireItemSlot(slotDir, 2, spinePlugin.itemSlotToken())).toBe(-1)
    expect(readFileSync(join(slotDir, 'slot-0'), 'utf8')).toBe(`${process.pid}:other`)
  })

  it('release frees only a slot that still holds the releasing token', () => {
    const slotDir = join(scratch(), 'item-slots')
    mkdirSync(slotDir, { recursive: true })
    const mine = spinePlugin.itemSlotToken()
    expect(spinePlugin.tryAcquireItemSlot(slotDir, 1, mine)).toBe(0)
    // Another claimant took the slot over (as a reclaim would): our late
    // release must leave its slot alone.
    writeFileSync(join(slotDir, 'slot-0'), `${process.pid}:theirs`)
    spinePlugin.releaseItemSlot(slotDir, 0, mine)
    expect(readFileSync(join(slotDir, 'slot-0'), 'utf8')).toBe(`${process.pid}:theirs`)
    spinePlugin.releaseItemSlot(slotDir, 0, `${process.pid}:theirs`)
    expect(existsSync(join(slotDir, 'slot-0'))).toBe(false)
  })

  it('a slot error ends the items instead of stranding the drain on its local permits', async () => {
    const items = Array.from({ length: 4 }, (_, i) => ({ id: `e${i}`, delay: 0 }))
    const origSlotsDir = process.env.REFACTORY_ITEM_SLOTS_DIR
    const origSlots = process.env.REFACTORY_ITEM_SLOTS
    // A slot directory that does not exist: every claim throws ENOENT.
    process.env.REFACTORY_ITEM_SLOTS_DIR = join(scratch(), 'missing', 'item-slots')
    process.env.REFACTORY_ITEM_SLOTS = '2'
    try {
      const run = loaderRun(items, 2)
      const ctx = await mountLoader(processShell([]), {
        graph: loaderGraph(2), workdir: run.workdir,
        commandEnv: { REFACTORY_WORKDIR: run.workdir, PYTHONPATH: run.taskDir },
      })
      const stranded = new Promise<string>(settle => setTimeout(() => settle('stranded'), 15_000))
      const outcome = await Promise.race([drive(ctx).then(state => String(state.status)), stranded])
      expect(outcome).not.toBe('stranded')
      expect(outcome).not.toBe('done')
    } finally {
      if (origSlotsDir === undefined) delete process.env.REFACTORY_ITEM_SLOTS_DIR
      else process.env.REFACTORY_ITEM_SLOTS_DIR = origSlotsDir
      if (origSlots === undefined) delete process.env.REFACTORY_ITEM_SLOTS
      else process.env.REFACTORY_ITEM_SLOTS = origSlots
    }
  }, 30_000)

})
