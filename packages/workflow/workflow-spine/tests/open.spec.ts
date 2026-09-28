/**
 * Opening a mode's graph after boot, owned by the session that opened it.
 *
 * The web chat session is one long-lived process, so the graph cannot be a
 * mount-time constant the way the `graph` profile makes it. These specs pin the
 * run-time-open composition: a stub open command lowers (here: pretends to lower)
 * a mode and prints where the graph is, `wf_open` claims the run for the calling
 * session, and every other `wf_*` tool works only on that session's run. The
 * configured-graph path is pinned too, so `interactiveOpen: false` stays exactly
 * what it was.
 */
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { afterEach, describe, expect, it } from 'vitest'
import * as spinePlugin from '../src/index.ts'
import type { Config } from '../src/index.ts'
import type { WorkflowGraph } from '../src/types.ts'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-spine-open-'))
  dirs.push(dir)
  return dir
}

/** An executable stub, so `spawnSync` runs it exactly as it would the console. */
function executable(body: string): string {
  const path = join(scratch(), 'open-stub')
  writeFileSync(path, `#!/bin/sh\n${body}\n`)
  chmodSync(path, 0o755)
  return path
}

/** One agent node, so the cursor has something a model must do. */
const AGENT_GRAPH: WorkflowGraph = {
  name: 't',
  start_node: 'a',
  nodes: { a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] } },
  edges: [],
}

/** One deterministic node, so `wf_next` runs it through the shell. */
const FN_GRAPH: WorkflowGraph = {
  name: 't',
  start_node: 'a',
  nodes: { a: { _type: 'FnNode', id: 'a', command: 'true', reads: [], writes: [] } },
  edges: [],
}

/**
 * Four deterministic nodes in a row, each with its own command and output.
 *
 * The four nodes are what make "one node per call" countable: a composition that
 * drains them all in one call runs four commands, one that stops after one runs a
 * single command per call.
 */
const CHAIN_GRAPH: WorkflowGraph = {
  name: 'chain',
  start_node: 'n1',
  nodes: {
    n1: { _type: 'FnNode', id: 'n1', command: 'run-n1', reads: [], writes: ['n1.out'] },
    n2: { _type: 'FnNode', id: 'n2', command: 'run-n2', reads: [], writes: ['n2.out'] },
    n3: { _type: 'FnNode', id: 'n3', command: 'run-n3', reads: [], writes: ['n3.out'] },
    n4: { _type: 'FnNode', id: 'n4', command: 'run-n4', reads: [], writes: ['n4.out'] },
  },
  edges: [
    { source: 'n1', target: 'n2' },
    { source: 'n2', target: 'n3' },
    { source: 'n3', target: 'n4' },
  ],
}

/** A deterministic start node, so `wf_open` owns one before a model is due. */
const PREP_GRAPH: WorkflowGraph = {
  name: 't',
  start_node: 'prep',
  nodes: {
    prep: { _type: 'FnNode', id: 'prep', command: 'prep', reads: [], writes: ['prep.out'] },
    a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
  },
  edges: [{ source: 'prep', target: 'a' }],
}

/**
 * A data node over two items whose subgraph entry is deterministic and whose exit
 * is model-owned, so one pass spans several calls.
 */
const DATA_GRAPH: WorkflowGraph = {
  name: 'dataset',
  start_node: 'data',
  nodes: {
    data: {
      _type: 'DataNode', id: 'data',
      task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'end',
      reads: [], writes: ['current_item.json'],
    },
    work: { _type: 'FnNode', id: 'work', command: 'work', reads: [], writes: [] },
    end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
    after: { _type: 'FnNode', id: 'after', command: 'after', reads: [], writes: [] },
  },
  edges: [
    { source: 'data', target: 'after' },
    { source: 'work', target: 'end' },
  ],
}

/** A lowered graph on disk, as `interactive.prepare` would leave it. */
function lower(run: string, graph: WorkflowGraph = AGENT_GRAPH): string {
  const path = join(run, 'graphs', 't.graph.json')
  mkdirSync(join(run, 'graphs'), { recursive: true })
  writeFileSync(path, JSON.stringify(graph))
  return path
}

/** The open command's payload, as the console's `open` verb returns it. */
function payload(run: string, graph: string, extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    mode: 'pkg/t',
    run,
    name: 't',
    graph,
    declarations: join(run, '.refactory', 'declarations.json'),
    ...extra,
  })
}

/** An executable that prints the payload `open` prints, for a run already lowered. */
function openStub(run: string, graph: string, log?: string, extra: Record<string, unknown> = {}): string {
  const line = log === undefined ? '' : `printf '%s\\n' "$@" > '${log}'\n`
  return executable(`${line}printf '%s' '${payload(run, graph, extra)}'`)
}

function config(overrides: Partial<Config> = {}): Config {
  return {
    toolPrefix: 'wf',
    maxIterations: 3,
    interactiveOpen: true,
    openCommand: '',
    openArgs: [],
    executeCommands: false,
    commandEnv: {},
    commandTimeoutMs: 10_000,
    commandSandboxMode: 'danger-full-access',
    dataCommand: '',
    steerCommand: '',
    steerArgs: [],
    ceoDeny: [],
    ...overrides,
  }
}

/** A context with the two services the plugin injects, and the plugin mounted. */
async function mount(overrides: Partial<Config> = {}, shell?: object): Promise<Context> {
  const root = new Context()
  await root.plugin(SystemPrompt)
  await root.plugin(ToolRuntime)
  if (shell !== undefined) root.provide('shell', shell)
  await root.plugin(spinePlugin, config(overrides))
  return root
}

/** A shell that records each command's environment and succeeds. */
function captureShell(seen: Array<Record<string, string> | undefined>): object {
  return {
    resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: 10_000,
      stdoutMaxBytes: 1_000_000,
      env: request.env,
    }),
    run: async (spec: { env?: Record<string, string> }) => {
      seen.push(spec.env)
      return {
        exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
        stdout: { text: '', truncated: false },
        stderr: { text: '', truncated: false },
      }
    },
  }
}

/**
 * A shell that records every command, answers a data node's loader verbs, and
 * leaves behind what a real node command leaves: its declared writes and the
 * run's `facts.json`.
 *
 * The record makes "one node per call" countable; the files make the `ran`
 * payload's outputs and facts real rather than declared.
 */
function stepShell(workdir: string, log: string[], items: Array<Record<string, unknown>> = []): object {
  return {
    resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
      command: request.command,
      workdir: request.workdir ?? workdir,
      timeoutMs: 10_000,
      stdoutMaxBytes: 1_000_000,
      env: request.env,
    }),
    run: async (spec: { command: string; workdir?: string; env?: Record<string, string> }) => {
      log.push(spec.command)
      const dir = spec.workdir ?? workdir
      const loader = spec.command.startsWith('python -m refactory_dsh.items')
      const verb = loader ? (spec.command.split(' ')[3] ?? '') : ''
      const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
      let stdout = ''
      if (verb === 'list') stdout = JSON.stringify({ count: items.length, ids: items.map(item => item.id) })
      else if (verb === 'begin') stdout = JSON.stringify(items[index] ?? {})
      else if (verb === 'end') stdout = JSON.stringify({ id: items[index]?.id, passed: true, score: 1 })
      else if (!loader) {
        // A node command: write the files it declared and fold a fact in, the way
        // `refactory_dsh.run` does.
        const manifest = JSON.parse(spec.env?.['DSH_WORKFLOW_NODE'] ?? '{}') as { id?: string; writes?: string[] }
        for (const rel of manifest.writes ?? []) writeFileSync(join(dir, rel), String(spec.command))
        const factsPath = join(dir, 'facts.json')
        let facts: Record<string, unknown> = {}
        try {
          facts = JSON.parse(readFileSync(factsPath, 'utf8')) as Record<string, unknown>
        } catch {
          facts = {}
        }
        facts[manifest.id ?? spec.command] = true
        writeFileSync(factsPath, JSON.stringify(facts))
      }
      return {
        exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
        stdout: { text: stdout, truncated: false },
        stderr: { text: '', truncated: false },
      }
    },
  }
}

/**
 * The execution facts a tool call is made with: the session that owns it. A real
 * Session always folds `request/header` events, so the stub answers `undefined`
 * (no header folded yet) rather than omitting the method the spine calls.
 */
function as(session: string): unknown {
  return { agent: { session: { id: session, requestHeader: () => undefined } } as unknown as Agent }
}

/**
 * The execution facts of a session whose model selection the run's workers
 * should inherit: the folded request header, exactly as a live session folds
 * `request/header` events, with the agent options as the fallback source.
 */
function asModel(session: string, selection: { provider?: string; model?: string } | undefined,
                 options?: { provider?: string; model?: string }): unknown {
  return {
    agent: {
      session: {
        id: session,
        requestHeader: () => (selection === undefined ? undefined : { config: selection }),
      },
      ...(options === undefined ? {} : { options }),
    } as unknown as Agent,
  }
}

async function call(ctx: Context, name: string, args: unknown = {}, exec: unknown = {}): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} not registered`)
  return definition.execute(args, exec as never)
}

describe('opening a graph at run time', () => {
  it('claims the run for the session and returns the start node', async () => {
    const run = scratch()
    const graph = lower(run)
    const ctx = await mount({ openCommand: openStub(run, graph) })

    const first = await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    expect(first).toMatchObject({ status: 'ready', nodes: [{ id: 'a' }] })

    // The claim is live: the same session's next call works on the opened run.
    expect(await call(ctx, 'wf_next', {}, as('session-1')))
      .toMatchObject({ status: 'ready', nodes: [{ id: 'a' }] })
  })

  it('passes the session model selection to the open command, from the header first', async () => {
    const run = scratch()
    const graph = lower(run)
    const log = join(scratch(), 'argv')
    const ctx = await mount({ openCommand: openStub(run, graph, log) })

    await call(ctx, 'wf_open', { mode: 'pkg/t', workdir: run }, asModel('session-1',
      { provider: 'litellm', model: 'rits/zai-org/glm-5-3' }, { provider: 'stale', model: 'stale' }))

    const argv = readFileSync(log, 'utf8').split('\n').filter(Boolean)
    expect(argv).toContain('--provider')
    expect(argv[argv.indexOf('--provider') + 1]).toBe('litellm')
    expect(argv[argv.indexOf('--model') + 1]).toBe('rits/zai-org/glm-5-3')
    // The agent's own options are the fallback when no header has been folded.
    const fallback = join(scratch(), 'argv')
    const ctx2 = await mount({ openCommand: openStub(run, graph, fallback) })
    await call(ctx2, 'wf_open', { mode: 'pkg/t', workdir: run }, asModel('session-2', undefined,
      { provider: 'litellm', model: 'rits/zai-org/glm-5-3' }))
    const fallbackArgv = readFileSync(fallback, 'utf8').split('\n').filter(Boolean)
    expect(fallbackArgv[fallbackArgv.indexOf('--provider') + 1]).toBe('litellm')
    expect(fallbackArgv[fallbackArgv.indexOf('--model') + 1]).toBe('rits/zai-org/glm-5-3')
    // A session with no selection at all passes neither flag, so the run keeps
    // the console's own defaults.
    const bare = join(scratch(), 'argv')
    const ctx3 = await mount({ openCommand: openStub(run, graph, bare) })
    await call(ctx3, 'wf_open', { mode: 'pkg/t', workdir: run }, as('session-3'))
    expect(readFileSync(bare, 'utf8')).not.toContain('--provider')
    expect(readFileSync(bare, 'utf8')).not.toContain('--model')
  })

  it('refuses a call before any run is open, naming the tool to call', async () => {
    const ctx = await mount({ openCommand: executable('exit 1') })
    await expect(call(ctx, 'wf_next', {}, as('session-1')))
      .rejects.toThrow('no mode is open in this session; call wf_open with a mode first')
  })

  it('refuses a second open from the same session, distinctly', async () => {
    const run = scratch()
    const graph = lower(run)
    const ctx = await mount({ openCommand: openStub(run, graph) })
    await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    await expect(call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1')))
      .rejects.toThrow('a run is already open in this session')
  })

  it('refuses a second open from another session, distinctly', async () => {
    const run = scratch()
    const graph = lower(run)
    const ctx = await mount({ openCommand: openStub(run, graph) })
    await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    await expect(call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-2')))
      .rejects.toThrow('a run is already open in another session')
  })

  it('refuses another session\'s tools on the opened run', async () => {
    const run = scratch()
    const graph = lower(run)
    const ctx = await mount({ openCommand: openStub(run, graph) })
    await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    for (const name of ['wf_next', 'wf_done', 'wf_state']) {
      await expect(call(ctx, name, { node: 'a' }, as('session-2')))
        .rejects.toThrow('a run is already open in another session')
    }
  })

  it('requires an owning agent session to open anything', async () => {
    const run = scratch()
    const graph = lower(run)
    const ctx = await mount({ openCommand: openStub(run, graph) })
    await expect(call(ctx, 'wf_open', { mode: 'pkg/t' }, {}))
      .rejects.toThrow('wf_open requires an owning agent session')
  })

  it('needs an open command to be configured', async () => {
    const ctx = await mount()
    await expect(call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1')))
      .rejects.toThrow(/needs an open command/)
  })

  it('passes the mode, the workdir and the knobs to the open command', async () => {
    const run = scratch()
    const graph = lower(run)
    const log = join(scratch(), 'argv.txt')
    const ctx = await mount({ openCommand: openStub(run, graph, log) })
    await call(ctx, 'wf_open', { mode: 'pkg/t', workdir: run, knobs: ['a=1', 'b=2'] }, as('session-1'))
    const argv = readFileSync(log, 'utf8').trim().split('\n')
    expect(argv).toEqual(['pkg/t', '--workdir', run, '--knob', 'a=1', '--knob', 'b=2'])
  })

  it('surfaces the open command\'s refusal rather than opening nothing', async () => {
    const ctx = await mount({ openCommand: executable("echo 'no such mode' >&2\nexit 2") })
    await expect(call(ctx, 'wf_open', { mode: 'pkg/ghost' }, as('session-1')))
      .rejects.toThrow(/could not lower pkg\/ghost: no such mode/)
  })

  it('derives and creates a per-session run directory when the caller names none', async () => {
    const run = scratch()
    const graph = lower(run)
    const log = join(scratch(), 'argv.txt')
    const ctx = await mount({ openCommand: openStub(run, graph, log) })
    // A unique session id, so the derived directory cannot collide with a real one.
    const session = `derived-${Date.now()}-${Math.random().toString(36).slice(2)}`
    const derived = join(process.cwd(), '.factory', 'web', session, 'pkg-t')
    try {
      await call(ctx, 'wf_open', { mode: 'pkg/t' }, as(session))
      const argv = readFileSync(log, 'utf8').trim().split('\n')
      expect(argv).toEqual(['pkg/t', '--workdir', derived])
      expect(existsSync(derived)).toBe(true)
    } finally {
      rmSync(join(process.cwd(), '.factory', 'web', session), { recursive: true, force: true })
    }
  })

  it('runs a node command in the opened run\'s own environment', async () => {
    const run = scratch()
    const graph = lower(run, FN_GRAPH)
    const seen: Array<Record<string, string> | undefined> = []
    const ctx = await mount(
      {
        // The composition's static environment is a decoy: the payload's must win.
        openCommand: openStub(run, graph, undefined, {
          commandEnv: {
            REFACTORY_WORKDIR: run,
            REFACTORY_PROJECT: '/project',
            REFACTORY_KNOBS: '{"step":1}',
            PATH: '/venv/bin',
          },
          dataCommand: 'python -m refactory_dsh.items',
        }),
        executeCommands: true,
        commandEnv: { REFACTORY_WORKDIR: '/wrong', REFACTORY_PROJECT: '/wrong' },
      },
      captureShell(seen),
    )

    await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    expect(await call(ctx, 'wf_next', {}, as('session-1'))).toMatchObject({ status: 'done' })

    const environment = seen.find(env => env?.DSH_WORKFLOW_NODE !== undefined)
    expect(environment?.REFACTORY_WORKDIR).toBe(run)
    expect(environment?.REFACTORY_PROJECT).toBe('/project')
    expect(environment?.REFACTORY_KNOBS).toBe('{"step":1}')
    expect(environment?.PATH).toBe('/venv/bin')
  })

  it('falls back to the configured environment when the payload omits one', async () => {
    const run = scratch()
    const graph = lower(run, FN_GRAPH)
    const seen: Array<Record<string, string> | undefined> = []
    const ctx = await mount(
      {
        openCommand: openStub(run, graph),
        executeCommands: true,
        commandEnv: { REFACTORY_WORKDIR: '/configured' },
        dataCommand: 'loader',
      },
      captureShell(seen),
    )

    await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    await call(ctx, 'wf_next', {}, as('session-1'))

    const environment = seen.find(env => env?.DSH_WORKFLOW_NODE !== undefined)
    expect(environment?.REFACTORY_WORKDIR).toBe('/configured')
  })
})

describe('one deterministic node per call when the run is opened interactively', () => {
  it('runs a single node per wf_next, with wf_open running the first', async () => {
    const run = scratch()
    const graph = lower(run, CHAIN_GRAPH)
    const log: string[] = []
    const ctx = await mount({ openCommand: openStub(run, graph), executeCommands: true }, stepShell(run, log))

    const first = await call(ctx, 'wf_open', { mode: 'pkg/t', workdir: run }, as('session-1'))
    expect(log).toEqual(['run-n1'])
    expect(first).toMatchObject({
      status: 'ready',
      nodes: [{ id: 'n2' }],
      ran: { node: 'n1', type: 'FnNode', outputs: ['n1.out'], facts: { n1: true } },
    })

    const second = await call(ctx, 'wf_next', {}, as('session-1'))
    expect(log).toEqual(['run-n1', 'run-n2'])
    expect(second).toMatchObject({ nodes: [{ id: 'n3' }], ran: { node: 'n2' } })

    const third = await call(ctx, 'wf_next', {}, as('session-1'))
    expect(log).toEqual(['run-n1', 'run-n2', 'run-n3'])
    expect(third).toMatchObject({ nodes: [{ id: 'n4' }], ran: { node: 'n3' } })

    const last = await call(ctx, 'wf_next', {}, as('session-1'))
    expect(log).toEqual(['run-n1', 'run-n2', 'run-n3', 'run-n4'])
    expect(last).toMatchObject({
      status: 'done',
      ran: { node: 'n4', outputs: ['n4.out'], facts: { n1: true, n2: true, n3: true, n4: true } },
    })
  })

  it('returns the post-drain view from wf_open, not the deterministic start node', async () => {
    const run = scratch()
    const graph = lower(run, PREP_GRAPH)
    const log: string[] = []
    const ctx = await mount({ openCommand: openStub(run, graph), executeCommands: true }, stepShell(run, log))

    const first = await call(ctx, 'wf_open', { mode: 'pkg/t', workdir: run }, as('session-1')) as {
      nodes: Array<{ id: string }>
      ran: { node: string }
    }
    expect(log).toEqual(['prep'])
    expect(first.ran.node).toBe('prep')
    // The start node ran, so it is not due; the agent node is what the model owes.
    expect(first.nodes.map(node => node.id)).toEqual(['a'])
  })

  it('keeps a data pass moving across calls without stranding it', async () => {
    const run = scratch()
    const graph = lower(run, DATA_GRAPH)
    const log: string[] = []
    const items = [{ id: 'a' }, { id: 'b' }]
    const ctx = await mount(
      { openCommand: openStub(run, graph), executeCommands: true, dataCommand: 'python -m refactory_dsh.items' },
      stepShell(run, log, items),
    )

    const first = await call(ctx, 'wf_open', { mode: 'pkg/t', workdir: run }, as('session-1')) as {
      item?: unknown
      ran?: { node: string }
    }
    expect(first.item).toEqual({ id: 'a' })
    expect(first.ran).toMatchObject({ node: 'work' })
    expect(log).toEqual([
      "python -m refactory_dsh.items list --node 'data'",
      "python -m refactory_dsh.items begin --node 'data' --index 0",
      'work',
    ])

    // The pass's exit is model-owned: it is handed over, with the item.
    const second = await call(ctx, 'wf_next', {}, as('session-1')) as {
      item?: unknown
      nodes: Array<{ id: string }>
    }
    expect(second.item).toEqual({ id: 'a' })
    expect(second.nodes.map(node => node.id)).toEqual(['end'])
    await call(ctx, 'wf_done', { node: 'end' }, as('session-1'))

    // The next call closes item a and begins item b: a pass that started continues.
    const third = await call(ctx, 'wf_next', {}, as('session-1')) as { item?: unknown; ran?: { node: string } }
    expect(third.item).toEqual({ id: 'b' })
    expect(third.ran).toMatchObject({ node: 'work' })
    await call(ctx, 'wf_done', { node: 'end' }, as('session-1'))

    const last = await call(ctx, 'wf_next', {}, as('session-1')) as { status: string; ran?: { node: string } }
    expect(last.status).toBe('done')
    expect(last.ran).toMatchObject({ node: 'after' })
    // `begin` runs once per item, not once per call that happens to be inside the pass.
    expect(log.filter(line => line.includes('items begin'))).toEqual([
      "python -m refactory_dsh.items begin --node 'data' --index 0",
      "python -m refactory_dsh.items begin --node 'data' --index 1",
    ])
  })
})

describe('the configured graph still behaves as before', () => {
  it('drains every deterministic node in one call and reports no single ran node', async () => {
    const run = scratch()
    const log: string[] = []
    const ctx = await mount(
      { interactiveOpen: false, graph: CHAIN_GRAPH, executeCommands: true, workdir: run },
      stepShell(run, log),
    )

    const view = await call(ctx, 'wf_next', {}) as { status: string; ran?: unknown }
    expect(log).toEqual(['run-n1', 'run-n2', 'run-n3', 'run-n4'])
    expect(view.status).toBe('done')
    expect(view.ran).toBeUndefined()
  })

  it('does not register wf_open and walks the configured graph', async () => {
    const ctx = await mount({ interactiveOpen: false, graph: AGENT_GRAPH })
    expect(ctx.tools.get('wf_open')).toBeUndefined()
    expect(await call(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'a' }] })
  })

  it('refuses a missing graph at mount when not opening at run time', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await expect(root.plugin(spinePlugin, config({ interactiveOpen: false })))
      .rejects.toThrow(/no workflow graph is configured/)
  })
})

describe('the status of an opened run', () => {
  it('is written under the opened run\'s workdir, not the configured one', async () => {
    // The configured workdir is a different directory on purpose: the status must
    // follow the run the session opened, not the composition's own.
    const configured = scratch()
    const run = scratch()
    const graph = lower(run)
    const ctx = await mount({ openCommand: openStub(run, graph), workdir: configured })

    await call(ctx, 'wf_open', { mode: 'pkg/t' }, as('session-1'))
    await call(ctx, 'wf_next', {}, as('session-1'))

    const written = join(run, '.refactory', 'spine-status.json')
    expect(existsSync(written)).toBe(true)
    expect(JSON.parse(readFileSync(written, 'utf8'))).toEqual({ status: 'ready', nodes: ['a'], completed: [] })
    expect(existsSync(join(configured, '.refactory', 'spine-status.json'))).toBe(false)
  })
})
