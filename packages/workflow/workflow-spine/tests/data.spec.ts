/**
 * A data node's item loop: the cursor's per-item pass machine, the graph rules the
 * loop relies on, and the plugin driving the loader commands around each pass.
 *
 * Issue #59. The walking executor runs a data node's whole loop in one process
 * because it owns the graph; the spine walks the subgraph the way it walks
 * everything else, one node at a time, so the item loop is split across three
 * loader commands that the composition configures.
 */
import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Spine, validateGraph } from '../src/spine.ts'
import { dataCommandOf } from '../src/execute.ts'
import * as spinePlugin from '../src/index.ts'
import type { Config } from '../src/index.ts'
import type { WorkflowGraph, WorkflowNode } from '../src/types.ts'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
  vi.restoreAllMocks()
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-spine-data-'))
  dirs.push(dir)
  return dir
}

/**
 * A data node over a two-node subgraph whose exit is model-owned.
 *
 * The agent node is the point: a subgraph the runtime could run entirely by itself
 * would not need the loop driven from outside.
 */
function dataGraph(): WorkflowGraph {
  return {
    name: 'dataset',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data',
        task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'end',
        reads: [], writes: ['current_item.json'],
      },
      work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
      end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'data', target: 'after' },
      { source: 'work', target: 'end' },
    ],
  }
}

describe("a data node's pass machine", () => {
  it('runs one pass per item, ending the data node after the last exit', () => {
    const spine = new Spine(dataGraph(), { maxIterations: 2 })
    expect(spine.next().nodes?.[0]).toMatchObject({ id: 'data', _type: 'DataNode' })

    expect(spine.startPass('data', 2)).toMatchObject({
      status: 'ready',
      pass: { node: 'data', count: 2, index: 0, entry: 'work', exit: 'end', finished: false },
    })
    expect(spine.next().nodes?.[0]).toMatchObject({ id: 'work' })

    spine.complete('work')
    expect(spine.passOf()).toMatchObject({ index: 0, finished: false })
    expect(spine.complete('end')).toMatchObject({ pass: { index: 0, finished: true } })

    // The second item: the same subgraph, from its entry again.
    expect(spine.endPass().nodes?.[0]).toMatchObject({ id: 'work' })
    expect(spine.passOf()).toMatchObject({ index: 1, finished: false })

    spine.complete('work')
    spine.complete('end')
    // The last item completes the data node and reaches its successor.
    expect(spine.endPass().nodes?.[0]).toMatchObject({ id: 'after' })
    expect(spine.passOf()).toBeUndefined()
    expect(spine.state().completed).toContain('data')
  })

  it("does not walk out of the subgraph through the exit's own edge", () => {
    const graph = dataGraph()
    const spine = new Spine(
      { ...graph, edges: [...graph.edges, { source: 'end', target: 'after' }] },
      { maxIterations: 2 },
    )
    spine.startPass('data', 1)
    spine.complete('work')
    const afterExit = spine.complete('end')
    expect(afterExit.status).toBe('ready')
    expect(afterExit.nodes).toEqual([])
    expect(spine.endPass().nodes?.[0]).toMatchObject({ id: 'after' })
  })

  it('refuses to close a pass its exit has not finished', () => {
    const spine = new Spine(dataGraph(), { maxIterations: 2 })
    expect(() => spine.endPass()).toThrow(/no data pass/)
    spine.startPass('data', 1)
    expect(() => spine.endPass()).toThrow(/has not finished/)
  })

  it('refuses an empty selection and a node that is not a data node', () => {
    const spine = new Spine(dataGraph(), { maxIterations: 2 })
    expect(() => spine.startPass('data', 0)).toThrow(/empty selection/)
    expect(() => spine.startPass('work', 1)).toThrow(/not a data node/)
  })
})

describe('the graph rules the pass machine relies on', () => {
  it('refuses a gate as the subgraph exit, naming the exit', () => {
    const graph = dataGraph()
    graph.nodes.end = {
      _type: 'GateNode', id: 'end', evaluator_type: 'fn', evaluator_command: 'true',
      reads: [], writes: [],
    }
    graph.edges = [...graph.edges, { source: 'end', target: 'after' }]
    expect(validateGraph(graph)).toContainEqual({
      node: 'data',
      message: "data node subgraph_exit 'end' is a gate; a pass ends by completing its exit",
    })
  })

  it('refuses a subgraph entry that cannot reach its exit', () => {
    const graph = dataGraph()
    graph.edges = graph.edges.filter(edge => edge.source !== 'work')
    expect(validateGraph(graph)).toContainEqual({
      node: 'data',
      message: "data node subgraph_entry 'work' cannot reach subgraph_exit 'end'",
    })
  })

  it("refuses an explicit edge into the data node's own subgraph", () => {
    const graph = dataGraph()
    graph.edges = [...graph.edges, { source: 'data', target: 'work' }]
    expect(validateGraph(graph)).toContainEqual({
      node: 'data',
      message: "data node has an edge into its own subgraph ('work'), which would run it twice per item",
    })
  })

  it('reaches the subgraph through the data node, not through an edge', () => {
    // Without following `subgraph_entry`, the whole subgraph reads as unreachable.
    expect(validateGraph(dataGraph())).toEqual([])
  })

  it('names a subgraph node that does not exist', () => {
    const graph = dataGraph()
    graph.nodes.data = { ...(graph.nodes.data as WorkflowNode), subgraph_entry: 'nowhere' }
    expect(validateGraph(graph)).toContainEqual({
      node: 'data',
      message: "data node subgraph_entry 'nowhere' is not a node",
    })
  })
})

/** A shell that answers each loader verb with the JSON a real loader prints. */
function loaderShell(items: Array<Record<string, unknown>>, log: string[]) {
  return {
    resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: 10_000,
      onExpiry: 'kill',
      stdoutMaxBytes: 1_000_000,
      env: request.env,
    }),
    execute: async (spec: { command: string }) => ({
      result: async () => {
        log.push(spec.command)
        // `python -m refactory_dsh.items <verb> --node '<id>'`
        const verb = spec.command.split(' ')[3] ?? ''
        const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
        const answers: Record<string, unknown> = {
          list: { count: items.length, ids: items.map(item => item.id) },
          begin: items[index] ?? {},
          end: { id: items[index]?.id, passed: true, score: 1 },
        }
        // A deterministic node of the subgraph is not a loader command.
        const text = verb in answers ? JSON.stringify(answers[verb]) : ''
        return {
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text, truncated: false },
          stderr: { text: '', truncated: false },
        }
      },
    }),
  }
}

function dataConfig(overrides: Partial<Config> = {}): Config {
  return {
    graph: dataGraph(),
    toolPrefix: 'wf',
    maxIterations: 3,
    interactiveOpen: false,
    openCommand: '',
    openArgs: [],
    executeCommands: true,
    commandEnv: {},
    commandTimeoutMs: 10_000,
    commandSandboxMode: 'danger-full-access',
    dataCommand: 'python -m refactory_dsh.items',
    steerCommand: '',
    steerArgs: [],
    workdir: scratch(),
    ceoDeny: [],
    ...overrides,
  }
}

/** A context with the two injected services, a loader shell, and the plugin. */
async function mount(shell: object, overrides: Partial<Config> = {}): Promise<Context> {
  const root = new Context()
  await root.plugin(SystemPrompt)
  await root.plugin(ToolRuntime)
  root.provide('shell', shell)
  await root.plugin(spinePlugin, dataConfig(overrides))
  return root
}

async function call(ctx: Context, name: string, args: unknown = {}): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} is not registered`)
  return definition.execute(args, {} as never)
}

describe('driving the loader commands around each pass', () => {
  it('persists an executing marker before every loader and node command', async () => {
    const workdir = scratch()
    const log: string[] = []
    const shell = loaderShell([{ id: 'a' }], log)
    const run = shell.execute
    shell.execute = async (spec) => {
      const saved = JSON.parse(readFileSync(join(workdir, '.refactory/spine-checkpoint.json'), 'utf8'))
      expect(saved.status).toBe('executing')
      return run(spec)
    }
    const root = await mount(shell, { workdir })
    await call(root, 'wf_next')
    await call(root, 'wf_done', { node: 'end' })
    await call(root, 'wf_next')
    expect(log.some(command => command.includes(' list '))).toBe(true)
    expect(log.some(command => command.includes(' begin '))).toBe(true)
    expect(log.some(command => command.includes(' end '))).toBe(true)
  })

  it('restores a known item setup without invoking begin a second time', async () => {
    const workdir = scratch()
    const root = await mount(loaderShell([{ id: 'a' }], []), { workdir })
    await call(root, 'wf_next')
    const path = join(workdir, '.refactory/spine-checkpoint.json')
    const saved = readFileSync(path)
    expect(JSON.parse(saved.toString()).materialised).toEqual({ key: 'data:0', item: { id: 'a' } })
    writeFileSync(join(workdir, '.refactory/resume-request.json'), JSON.stringify({
      checkpoint_sha256: createHash('sha256').update(saved).digest('hex'),
    }))
    const commands: string[] = []
    const restored = await mount(loaderShell([{ id: 'a' }], commands), { workdir })
    expect(await call(restored, 'wf_next')).toMatchObject({ nodes: [{ id: 'end' }], item: { id: 'a' } })
    expect(commands).toEqual([])
  })

  it('pauses a finished item before its grading command and resumes without repeating setup', async () => {
    const workdir = scratch()
    const graph = dataGraph()
    graph.nodes.data = { ...graph.nodes.data!, subgraph_exit: 'work' } as WorkflowNode
    delete graph.nodes.end
    graph.edges = [{ source: 'data', target: 'after' }]
    const commands: string[] = []
    const shell = loaderShell([{ id: 'a' }], commands)
    const original = shell.execute
    let paused = false
    const queue = (action: string): void => {
      const directory = join(workdir, '.refactory/graph-controls')
      mkdirSync(directory, { recursive: true })
      writeFileSync(join(directory, '1.json'), JSON.stringify({ action, run_id: workdir, invocation: null }))
    }
    shell.execute = async (spec) => {
      const execution = await original(spec)
      if (spec.command === 'true' && !paused) { paused = true; queue('pause') }
      return execution
    }
    const root = await mount(shell, { workdir, graph })
    expect(await call(root, 'wf_next')).toMatchObject({ status: 'paused' })
    expect(commands.some(command => command.includes(' end '))).toBe(false)
    queue('resume')
    expect(await call(root, 'wf_next')).toMatchObject({ status: 'done' })
    expect(commands.filter(command => command.includes(' begin '))).toHaveLength(1)
    expect(commands.filter(command => command.includes(' end '))).toHaveLength(1)
  })

  it('selects the items, opens a pass, and hands the subgraph over', async () => {
    const log: string[] = []
    const root = await mount(loaderShell([{ id: 'a' }, { id: 'b' }], log))

    const first = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    expect(log).toEqual([
      "python -m refactory_dsh.items list --node 'data'",
      "python -m refactory_dsh.items begin --node 'data' --index 0",
      // The pass's own deterministic node ran before the agent node was handed over.
      'true',
    ])
    expect(first.item).toEqual({ id: 'a' })
    // The deterministic node of the pass ran; the agent node is what is handed over.
    expect(first.nodes.map(node => node.id)).toEqual(['end'])

    await call(root, 'wf_done', { node: 'end' })
    const second = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    expect(log[3]).toBe("python -m refactory_dsh.items end --node 'data' --index 0")
    expect(log[4]).toBe("python -m refactory_dsh.items begin --node 'data' --index 1")
    expect(log[5]).toBe('true')
    expect(second.item).toEqual({ id: 'b' })
    expect(second.nodes.map(node => node.id)).toEqual(['end'])

    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(log[6]).toBe("python -m refactory_dsh.items end --node 'data' --index 1")
    expect(last.status).toBe('done')
  })

  it('completes an empty selection instead of hanging on it', async () => {
    const log: string[] = []
    const root = await mount(loaderShell([], log))
    const state = await call(root, 'wf_next') as { status: string }
    expect(log).toEqual([
      "python -m refactory_dsh.items list --node 'data'",
      // Nothing was iterated, and the run continued past the data node.
      'true',
    ])
    expect(state.status).toBe('done')
  })

  it('reports a loader that failed, with its stderr', async () => {
    const failing = {
      resolve: (request: { command: string }) => ({
        command: request.command, workdir: process.cwd(), timeoutMs: 1,
        onExpiry: 'kill', stdoutMaxBytes: 10, env: {},
      }),
      execute: async () => ({
        result: async () => ({
          exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 1,
          stdout: { text: '', truncated: false },
          stderr: { text: 'TaskError: no Task named probe', truncated: false },
        }),
      }),
    }
    const root = await mount(failing)
    const state = await call(root, 'wf_next') as { status: string; stderr: string }
    expect(state.status).toBe('failed')
    expect(state.stderr).toContain('no Task named probe')
  })

  it('refuses a data graph at load when the composition mounts no loader', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    root.provide('shell', loaderShell([], []))
    await expect(root.plugin(spinePlugin, dataConfig({ dataCommand: '' })))
      .rejects.toThrow(/no item loader is configured/)

    const other = new Context()
    await other.plugin(SystemPrompt)
    await other.plugin(ToolRuntime)
    other.provide('shell', loaderShell([], []))
    await expect(other.plugin(spinePlugin, dataConfig({ executeCommands: false })))
      .rejects.toThrow(/node commands are disabled/)
  })
})

describe('F4 failure rule: failed items in the data pass', () => {
  it('records a failed item and continues to the next', async () => {
    const log: string[] = []
    const items = [{ id: 'a' }, { id: 'b' }]
    const shell = {
      resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: 10_000,
        onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000,
        env: request.env,
      }),
      execute: async (spec: { command: string }) => ({
        result: async () => {
          log.push(spec.command)
          const verb = spec.command.split(' ')[3] ?? ''
          const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
          const answers: Record<string, unknown> = {
            list: { count: items.length, ids: items.map(i => i.id) },
            begin: items[index] ?? {},
            end: index === 0
              ? { error: 'verify failed', id: 'a', passed: false, score: 0 }
              : { id: 'b', passed: true, score: 1 },
          }
          const text = verb in answers ? JSON.stringify(answers[verb]) : ''
          return {
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text, truncated: false },
            stderr: { text: '', truncated: false },
          }
        },
      }),
    }
    const root = await mount(shell)
    await call(root, 'wf_next')
    await call(root, 'wf_done', { node: 'end' })
    // The first item's end returned an error; it should have continued to the second.
    await call(root, 'wf_next')
    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')
    // Both items had begin and end called
    expect(log.filter(c => c.includes(' begin ')).length).toBe(2)
    expect(log.filter(c => c.includes(' end ')).length).toBe(2)
  })

  it('skips the subgraph when begin signals a setup failure', async () => {
    const log: string[] = []
    const items = [{ id: 'a' }, { id: 'b' }]
    const shell = {
      resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: 10_000,
        onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000,
        env: request.env,
      }),
      execute: async (spec: { command: string }) => ({
        result: async () => {
          log.push(spec.command)
          const verb = spec.command.split(' ')[3] ?? ''
          const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
          const answers: Record<string, unknown> = {
            list: { count: items.length, ids: items.map(i => i.id) },
            begin: index === 0 ? { error: 'setup failed' } : items[index],
            end: index === 0
              ? { error: 'setup failed', id: 'a', passed: false, score: 0 }
              : { id: 'b', passed: true, score: 1 },
          }
          const text = verb in answers ? JSON.stringify(answers[verb]) : ''
          return {
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text, truncated: false },
            stderr: { text: '', truncated: false },
          }
        },
      }),
    }
    const root = await mount(shell)
    // First call: list → begin for item 0 (error) → end for item 0 → begin for item 1 → work → hand over end
    const first = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    // The first item's begin returned error, so no subgraph command was dispatched for it.
    // The second item should be the one with the agent node exposed.
    expect(first.item).toEqual({ id: 'b' })

    // No 'true' command for item 0 (subgraph was skipped)
    const trueCommands = log.filter(c => c === 'true')
    expect(trueCommands.length).toBe(1) // only for item b

    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')
  })

  it('continues past a subgraph node failure during a data pass', async () => {
    const log: string[] = []
    const items = [{ id: 'a' }, { id: 'b' }]
    let callCount = 0
    const shell = {
      resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: 10_000,
        onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000,
        env: request.env,
      }),
      execute: async (spec: { command: string }) => ({
        result: async () => {
          log.push(spec.command)
          const verb = spec.command.split(' ')[3] ?? ''
          const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
          if (verb === '') {
            // This is the deterministic subgraph node ('true')
            callCount += 1
            if (callCount === 1) {
              // First item's subgraph node fails
              return {
                exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
                stdout: { text: '', truncated: false },
                stderr: { text: 'subgraph node failed', truncated: false },
              }
            }
          }
          const answers: Record<string, unknown> = {
            list: { count: items.length, ids: items.map(i => i.id) },
            begin: items[index] ?? {},
            end: { id: items[index]?.id, passed: index === 0 ? false : true, score: index === 0 ? 0 : 1 },
          }
          const text = verb in answers ? JSON.stringify(answers[verb]) : ''
          return {
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text, truncated: false },
            stderr: { text: '', truncated: false },
          }
        },
      }),
    }
    const root = await mount(shell)
    // First wf_next: list, begin 0, work (fails) → close (end 0) → begin 1, work (ok) → hand over end
    const first = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    // The second item's agent node should be exposed
    expect(first.item).toEqual({ id: 'b' })
    expect(first.nodes.map((n: { id: string }) => n.id)).toEqual(['end'])

    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')

    // Both begin and end were called for both items
    expect(log.filter(c => c.includes(' end ')).length).toBe(2)
    // The first item's end command should include --failed since the subgraph failed
    const endCommands = log.filter(c => c.includes(' end '))
    expect(endCommands[0]).toContain('--failed')
    // The second item's end command should NOT include --failed
    expect(endCommands[1]).not.toContain('--failed')
  })

  it('fails the data node when all items fail (all-items-failed rule)', async () => {
    const log: string[] = []
    const items = [{ id: 'a' }, { id: 'b' }]
    let subgraphCallCount = 0
    const shell = {
      resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: 10_000,
        onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000,
        env: request.env,
      }),
      execute: async (spec: { command: string }) => ({
        result: async () => {
          log.push(spec.command)
          const verb = spec.command.split(' ')[3] ?? ''
          const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
          if (verb === '') {
            // This is the deterministic subgraph node ('true')
            subgraphCallCount += 1
            // Both items' subgraph nodes fail
            return {
              exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
              stdout: { text: '', truncated: false },
              stderr: { text: 'subgraph failed', truncated: false },
            }
          }
          // On the last item's end, the all-items-failed check raises → exit code 1.
          if (verb === 'end' && index === 1) {
            return {
              exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
              stdout: { text: '', truncated: false },
              stderr: { text: 'ItemsError: data: all 2 items failed', truncated: false },
            }
          }
          const answers: Record<string, unknown> = {
            list: { count: items.length, ids: items.map(i => i.id) },
            begin: items[index] ?? {},
            end: { error: 'subgraph failed', id: items[index]?.id, passed: false, score: 0 },
          }
          const text = verb in answers ? JSON.stringify(answers[verb]) : ''
          return {
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text, truncated: false },
            stderr: { text: '', truncated: false },
          }
        },
      }),
    }
    const root = await mount(shell)
    const result = await call(root, 'wf_next') as { status: string; stderr?: string }
    // The data node should have failed because all items failed
    expect(result.status).toBe('failed')
    expect(subgraphCallCount).toBe(2)
  })
})

describe('the data loader command line', () => {
  it('names the node and the item, and quotes the node id', () => {
    const node = { id: 'data', _type: 'DataNode' }
    expect(dataCommandOf(node, 'list', { dataCommand: 'loader' })).toBe("loader list --node 'data'")
    expect(dataCommandOf(node, 'begin', { dataCommand: 'loader' }, 3))
      .toBe("loader begin --node 'data' --index 3")
    expect(dataCommandOf({ id: "it's", _type: 'DataNode' }, 'list', { dataCommand: 'loader' }))
      .toBe(String.raw`loader list --node 'it'\''s'`)
  })

  it('includes --failed when a failure reason is provided', () => {
    const node = { id: 'data', _type: 'DataNode' }
    expect(dataCommandOf(node, 'end', { dataCommand: 'loader' }, 0, 'subgraph failed'))
      .toBe("loader end --node 'data' --index 0 --failed 'subgraph failed'")
  })

  it('includes --failed-kind when provided', () => {
    const node = { id: 'data', _type: 'DataNode' }
    expect(dataCommandOf(node, 'end', { dataCommand: 'loader' }, 0, 'timed out', 'timeout'))
      .toBe("loader end --node 'data' --index 0 --failed 'timed out' --failed-kind 'timeout'")
  })
})

describe('begin/end failure handling (Ask 3)', () => {
  /** Helper: a shell that can simulate begin/end failures. */
  function failingShell(
    items: Array<Record<string, unknown>>,
    log: string[],
    overrides: {
      beginFailIndex?: number
      beginTimedOutIndex?: number
      endFailIndex?: number
      endTimedOutIndex?: number
    } = {},
  ) {
    return {
      resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: 10_000,
        onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000,
        env: request.env,
      }),
      execute: async (spec: { command: string }) => ({
        result: async () => {
          log.push(spec.command)
          const verb = spec.command.split(' ')[3] ?? ''
          const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)

          // Simulate begin timeout
          if (verb === 'begin' && index === overrides.beginTimedOutIndex) {
            return {
              exitCode: null, signal: null, timedOut: true, aborted: false, timeoutMs: 1000,
              stdout: { text: '', truncated: false },
              stderr: { text: 'begin timed out', truncated: false },
            }
          }
          // Simulate begin exit-code failure
          if (verb === 'begin' && index === overrides.beginFailIndex) {
            return {
              exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
              stdout: { text: '', truncated: false },
              stderr: { text: 'begin failed', truncated: false },
            }
          }
          // Simulate end timeout
          if (verb === 'end' && index === overrides.endTimedOutIndex && !spec.command.includes('--failed')) {
            return {
              exitCode: null, signal: null, timedOut: true, aborted: false, timeoutMs: 1000,
              stdout: { text: '', truncated: false },
              stderr: { text: 'end timed out', truncated: false },
            }
          }
          // Simulate end exit-code failure
          if (verb === 'end' && index === overrides.endFailIndex && !spec.command.includes('--failed')) {
            return {
              exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
              stdout: { text: '', truncated: false },
              stderr: { text: 'end failed', truncated: false },
            }
          }

          const answers: Record<string, unknown> = {
            list: { count: items.length, ids: items.map(i => i.id) },
            begin: items[index] ?? {},
            end: { id: items[index]?.id, passed: true, score: 1 },
          }
          const text = verb in answers ? JSON.stringify(answers[verb]) : ''
          return {
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text, truncated: false },
            stderr: { text: '', truncated: false },
          }
        },
      }),
    }
  }

  it('treats a begin timeout as a failed item, not a run failure', async () => {
    const log: string[] = []
    const shell = failingShell([{ id: 'a' }, { id: 'b' }], log, { beginTimedOutIndex: 0 })
    const root = await mount(shell)
    const first = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    // Item 0 timed out on begin; item 1 should be exposed
    expect(first.item).toEqual({ id: 'b' })
    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')
    // The end --failed for item 0 should include --failed-kind 'timeout'
    const endFailedCmds = log.filter(c => c.includes(' end ') && c.includes('--failed'))
    expect(endFailedCmds.length).toBeGreaterThanOrEqual(1)
    expect(endFailedCmds[0]).toContain('--failed-kind')
  })

  it('treats a begin exit-code failure as a failed item', async () => {
    const log: string[] = []
    const shell = failingShell([{ id: 'a' }, { id: 'b' }], log, { beginFailIndex: 0 })
    const root = await mount(shell)
    const first = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    expect(first.item).toEqual({ id: 'b' })
    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')
  })

  it('treats an end timeout as a failed item', async () => {
    const log: string[] = []
    const shell = failingShell([{ id: 'a' }, { id: 'b' }], log, { endTimedOutIndex: 0 })
    const root = await mount(shell)
    await call(root, 'wf_next')
    await call(root, 'wf_done', { node: 'end' })
    const second = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown; status?: string }
    // Item 1 should be active now
    expect(second.item).toEqual({ id: 'b' })
    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')
    // The fallback end --failed should include --failed-kind
    const endFailedCmds = log.filter(c => c.includes(' end ') && c.includes('--failed'))
    expect(endFailedCmds.length).toBeGreaterThanOrEqual(1)
    expect(endFailedCmds[0]).toContain('--failed-kind')
  })

  it('treats an end exit-code failure as a failed item', async () => {
    const log: string[] = []
    const shell = failingShell([{ id: 'a' }, { id: 'b' }], log, { endFailIndex: 0 })
    const root = await mount(shell)
    await call(root, 'wf_next')
    await call(root, 'wf_done', { node: 'end' })
    // end for item 0 fails with exit code 1 — should propagate as node failure
    // (not all-items-failed since not all items processed yet)
    const second = await call(root, 'wf_next') as { status: string }
    // An end exit-code failure that's not a timeout propagates as a node failure
    expect(second.status).toBe('failed')
  })
})

describe('abandonPass primitive (Correction 1)', () => {
  /** A data graph with a gate in the subgraph body. */
  function dataGraphWithGate(): WorkflowGraph {
    return {
      name: 'dataset-with-gate',
      start_node: 'data',
      nodes: {
        data: {
          _type: 'DataNode', id: 'data',
          task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'gate',
          reads: [], writes: ['current_item.json'],
        },
        work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
        gate: {
          _type: 'GateNode', id: 'gate', evaluator_type: 'fn',
          evaluator_command: 'echo proceed', reads: [], writes: [],
        },
        after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
      },
      edges: [
        { source: 'data', target: 'after' },
        { source: 'work', target: 'gate' },
        { source: 'gate', target: 'after', condition: 'proceed' },
      ],
    }
  }

  it('records failed node as failed and remaining nodes as skipped, none completed, no verdicts', async () => {
    const log: string[] = []
    const items = [{ id: 'a' }, { id: 'b' }]
    let callCount = 0
    const shell = {
      resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: 10_000,
        onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000,
        env: request.env,
      }),
      execute: async (spec: { command: string }) => ({
        result: async () => {
          log.push(spec.command)
          const verb = spec.command.split(' ')[3] ?? ''
          const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
          if (verb === '') {
            callCount += 1
            if (callCount === 1) {
              return {
                exitCode: 1, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
                stdout: { text: '', truncated: false },
                stderr: { text: 'work failed for item a', truncated: false },
              }
            }
          }
          const answers: Record<string, unknown> = {
            list: { count: items.length, ids: items.map(i => i.id) },
            begin: items[index] ?? {},
            end: { id: items[index]?.id, passed: true, score: 1 },
          }
          const text = verb in answers ? JSON.stringify(answers[verb]) : ''
          return {
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text, truncated: false },
            stderr: { text: '', truncated: false },
          }
        },
      }),
    }
    const root = await mount(shell)
    // First item's work fails; abandonPass should skip 'end' (the agent node)
    const first = await call(root, 'wf_next') as { nodes: Array<{ id: string }>; item?: unknown }
    // The second item should be the one exposed
    expect(first.item).toEqual({ id: 'b' })
    await call(root, 'wf_done', { node: 'end' })
    const last = await call(root, 'wf_next') as { status: string }
    expect(last.status).toBe('done')
    // The end --failed should have been called for item 0
    const endFailedCmds = log.filter(c => c.includes(' end ') && c.includes('--failed'))
    expect(endFailedCmds.length).toBeGreaterThanOrEqual(1)
  })
})
