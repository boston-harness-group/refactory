/**
 * The spine plugin's composition: graph rejection at load, the cursor status it
 * leaves on disk, executing deterministic nodes, the tool renders, and the guard
 * that takes the CEO's coding tools away.
 *
 * These are the parts a run depends on but a single tool call does not reach.
 */
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawn, spawnSync } from 'node:child_process'
import { Context } from '@deepseek-ai/cordis'
import { createScope } from '@deepseek-ai/dsh-scope'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import UserQuestionService from '@deepseek-ai/dsh-user-questions'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { afterEach, describe, expect, it, vi, type MockInstance } from 'vitest'
import * as spinePlugin from '../src/index.ts'
import type { Config } from '../src/index.ts'
import type { WorkflowGraph } from '../src/types.ts'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
  vi.restoreAllMocks()
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-spine-composition-'))
  dirs.push(dir)
  return dir
}

/** One agent node, so the cursor has something a model must do. */
const AGENT_GRAPH: WorkflowGraph = {
  name: 't',
  start_node: 'a',
  nodes: { a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] } },
  edges: [],
}

function config(overrides: Partial<Config> = {}): Config {
  return {
    graph: AGENT_GRAPH,
    toolPrefix: 'wf',
    maxIterations: 3,
    interactiveOpen: false,
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
async function mount(overrides: Partial<Config> = {}): Promise<{ ctx: Context; tools: ToolRuntime }> {
  const root = new Context()
  await root.plugin(SystemPrompt)
  await root.plugin(ToolRuntime)
  await root.plugin(spinePlugin, config(overrides))
  return { ctx: root, tools: root.tools }
}

/** The shell request a node command is resolved from, as the plugin builds it. */
interface ShellRequest {
  command: string
  workdir?: string
  timeoutMs?: number
  env?: Record<string, string>
  sandboxPolicy?: unknown
}

/** A shell that reports whatever the caller asks for. */
function shell(behaviour: (command: string) => { exitCode?: number; stdout?: string; stderr?: string } = () => ({ exitCode: 0 })): object {
  return {
    resolve: (request: ShellRequest) => ({
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: request.timeoutMs ?? 10_000,
      onExpiry: 'kill',
      stdoutMaxBytes: 1_000_000,
      env: request.env,
      sandboxPolicy: request.sandboxPolicy,
    }),
    execute: async (spec: { command: string }) => ({
      result: async () => {
        const result = behaviour(spec.command)
        return {
          exitCode: result.exitCode ?? 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text: result.stdout ?? '', truncated: false },
          stderr: { text: result.stderr ?? '', truncated: false },
        }
      },
    }),
  }
}

async function call(ctx: Context, name: string, args: unknown = {}): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} not registered`)
  return definition.execute(args, {} as never)
}

/** Real asynchronous shell I/O for the advisory lifecycle regression. */
function processShell(): object {
  return {
    resolve: (request: ShellRequest) => ({ ...request, workdir: request.workdir ?? process.cwd(), timeoutMs: 1_000 }),
    execute: async (spec: ShellRequest) => {
      const child = spawn('/bin/sh', ['-c', spec.command], { cwd: spec.workdir, env: { ...process.env, ...spec.env } })
      let stdout = '', stderr = '', timedOut = false
      child.stdout.on('data', bytes => { stdout += String(bytes) })
      child.stderr.on('data', bytes => { stderr += String(bytes) })
      const timer = setTimeout(() => { timedOut = true; child.kill('SIGTERM') }, spec.timeoutMs ?? 1_000)
      const result = new Promise(resolve => child.on('close', (exitCode, signal) => {
        clearTimeout(timer)
        resolve({ exitCode, signal, timedOut, aborted: false, timeoutMs: spec.timeoutMs,
          stdout: { text: stdout, truncated: false }, stderr: { text: stderr, truncated: false } })
      }))
      return { result: async () => result }
    },
  }
}

describe('owned advisory commands', () => {
  it('starts the builder before archive settlement and records unfinished ownership', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = { name: 'advisory', start_node: 'archive', nodes: {
      archive: { id: 'archive', _type: 'FnNode', blocking: false,
        command: 'while [ ! -f builder-started ]; do sleep 0.02; done; printf archived > archive.txt' },
      builder: { id: 'builder', _type: 'FnNode', command: 'cp .refactory/spine-checkpoint.json builder-checkpoint.json; touch builder-started' },
    }, edges: [{ source: 'archive', target: 'builder' }] }
    const { ctx } = await mount({ graph, executeCommands: true, workdir })
    ctx.provide('shell', processShell() as never)
    await expect(call(ctx, 'wf_next')).resolves.toMatchObject({ status: 'done' })
    expect(readFileSync(join(workdir, 'archive.txt'), 'utf8')).toBe('archived')
    expect(JSON.parse(readFileSync(join(workdir, 'builder-checkpoint.json'), 'utf8')).advisory_pending).toEqual(['archive'])
    expect(JSON.parse(readFileSync(join(workdir, '.refactory/spine-status.json'), 'utf8')).advisory_pending).toEqual([])
    await ctx.fiber.dispose()
  })

  it('reports archive failure even after the builder completes', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = { name: 'advisory', start_node: 'archive', nodes: {
      archive: { id: 'archive', _type: 'FnNode', blocking: false, command: 'sleep 0.05; exit 7' },
      builder: { id: 'builder', _type: 'FnNode', command: 'touch builder-completed' },
    }, edges: [{ source: 'archive', target: 'builder' }] }
    const { ctx } = await mount({ graph, executeCommands: true, workdir })
    ctx.provide('shell', processShell() as never)
    await expect(call(ctx, 'wf_next')).resolves.toMatchObject({ status: 'failed', exit_code: 7 })
    expect(readFileSync(join(workdir, 'builder-completed'), 'utf8')).toBe('')
    expect(JSON.parse(readFileSync(join(workdir, '.refactory/spine-status.json'), 'utf8')).status).toBe('failed')
    expect(JSON.parse(readFileSync(join(workdir, '.refactory/spine-status.json'), 'utf8')).completed).not.toContain('archive')
    await ctx.fiber.dispose()
  })
})

describe('mounting a graph', () => {
  it('refuses an invalid graph at load, naming the problem', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await expect(root.plugin(spinePlugin, config({
      graph: { name: 't', start_node: 'missing', nodes: {}, edges: [] },
    }))).rejects.toThrow(/invalid workflow graph:/)
  })

  it('names the node an issue belongs to', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await expect(root.plugin(spinePlugin, config({
      graph: {
        name: 't',
        start_node: 'a',
        nodes: {
          a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
          ghost: { _type: 'AgentNode', id: 'ghost', role: 'builder', prompt_template: 'G', writes: [], reads: [] },
        },
        edges: [],
      },
    }))).rejects.toThrow(/ghost: unreachable from start_node/)
  })
})

describe('the cursor status a run leaves behind', () => {
  it('records ready while work is due and done once it is not', async () => {
    const workdir = scratch()
    const { ctx } = await mount({ workdir })
    await call(ctx, 'wf_next')
    expect(JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')))
      .toMatchObject({ status: 'ready', nodes: ['a'], completed: [] })

    await call(ctx, 'wf_done', { node: 'a' })
    await call(ctx, 'wf_next')
    expect(JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')))
      .toMatchObject({ status: 'done', completed: ['a'], nodes: [] })
  })

  it('records a refusal, with the reason, when no answerer can be asked', async () => {
    const workdir = scratch()
    const { ctx } = await mount({
      workdir,
      graph: {
        name: 't',
        start_node: 'gate',
        nodes: { gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] } },
        edges: [{ source: 'gate', target: 'gate', condition: 'proceed' }],
      },
    })
    await expect(call(ctx, 'wf_verdict', { gate: 'gate' }))
      .rejects.toThrow(/mounts no user-questions channel/)
    const recorded = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')) as { status: string; error: string }
    expect(recorded.status).toBe('refused')
    expect(recorded.error).toContain('gate gate')
  })

  it('records the failed node\'s own stderr, not only its exit code', async () => {
    // "exit code 4" sends a reader back to re-run the node to find out why; the
    // stderr is the reason, and the run directory is where a reader looks.
    const workdir = scratch()
    const { ctx } = await mount({
      workdir,
      executeCommands: true,
      graph: {
        name: 't',
        start_node: 'run',
        nodes: { run: { _type: 'FnNode', id: 'run', command: 'run-it', reads: [], writes: [] } },
        edges: [],
      },
    })
    ctx.provide('shell', shell(() => ({ exitCode: 4, stderr: 'broken on purpose\n' })))
    await expect(call(ctx, 'wf_next')).resolves.toMatchObject({ status: 'failed', exit_code: 4 })
    const recorded = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')) as { status: string; error: string }
    expect(recorded.status).toBe('failed')
    expect(recorded.error).toContain('broken on purpose')
  })

  it('records the exit code alone when a failed node said nothing', async () => {
    const workdir = scratch()
    const { ctx } = await mount({
      workdir,
      executeCommands: true,
      graph: {
        name: 't',
        start_node: 'run',
        nodes: { run: { _type: 'FnNode', id: 'run', command: 'run-it', reads: [], writes: [] } },
        edges: [],
      },
    })
    ctx.provide('shell', shell(() => ({ exitCode: 4 })))
    await expect(call(ctx, 'wf_next')).resolves.toMatchObject({ status: 'failed' })
    const recorded = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')) as { error: string }
    expect(recorded.error).toBe('run exited with code 4')
  })

  it('refuses dispatch when the recovery checkpoint cannot be written', async () => {
    const blocked = join(scratch(), 'not-a-directory')
    writeFileSync(blocked, 'x')
    const { ctx } = await mount({ workdir: join(blocked, 'child'), executeCommands: true,
      graph: { name: 'blocked', start_node: 'a', nodes: { a: { id: 'a', _type: 'FnNode', command: 'never' } }, edges: [] } })
    const executed = vi.fn(() => ({ exitCode: 0 }))
    ctx.provide('shell', shell(executed))
    await expect(call(ctx, 'wf_next')).rejects.toThrow(/Cannot persist graph recovery checkpoint/)
    expect(executed).not.toHaveBeenCalled()
  })
})

describe('deterministic nodes', () => {
  it.each(['pause', 'stop'])('applies queued %s before dispatching the next node in the same drain', async (action) => {
    const workdir = scratch()
    const { ctx } = await mount({ workdir, executeCommands: true, graph: {
      name: 'boundaries', start_node: 'a', nodes: {
        a: { id: 'a', _type: 'FnNode', command: 'first' },
        b: { id: 'b', _type: 'FnNode', command: 'second' },
      }, edges: [{ source: 'a', target: 'b' }],
    } })
    const commands: string[] = []
    ctx.provide('shell', shell(command => {
      commands.push(command)
      const directory = join(workdir, '.refactory/graph-controls')
      mkdirSync(directory, { recursive: true })
      writeFileSync(join(directory, '1.json'), JSON.stringify({ action, run_id: workdir, invocation: null }))
      return { exitCode: 0 }
    }))
    await call(ctx, 'wf_next')
    expect(commands).toEqual(['first'])
    const saved = JSON.parse(readFileSync(join(workdir, '.refactory/spine-checkpoint.json'), 'utf8'))
    expect(saved.state.completed).toEqual(['a'])
    expect(saved.status).toBe(action === 'pause' ? 'paused' : 'halted')
  })

  it('hands the node over when no shell is mounted', async () => {
    const { ctx } = await mount({
      executeCommands: true,
      graph: {
        name: 't',
        start_node: 'run',
        nodes: { run: { _type: 'FnNode', id: 'run', command: 'true', reads: [], writes: [] } },
        edges: [],
      },
    })
    await expect(call(ctx, 'wf_next')).resolves.toMatchObject({ status: 'ready', nodes: [{ id: 'run' }] })
  })

  it('runs the last deterministic node and reports done', async () => {
    const { ctx } = await mount({
      workdir: scratch(),
      executeCommands: true,
      graph: {
        name: 't',
        start_node: 'run',
        nodes: { run: { _type: 'FnNode', id: 'run', command: 'true', reads: [], writes: [] } },
        edges: [],
      },
    })
    ctx.provide('shell', shell())
    await expect(call(ctx, 'wf_next')).resolves.toMatchObject({ status: 'done' })
  })
})

describe('tool presentation', () => {
  it('renders every tool\'s value as compact JSON text', async () => {
    const { ctx } = await mount()
    for (const name of ['wf_next', 'wf_done', 'wf_verdict', 'wf_edit', 'wf_expand', 'wf_state']) {
      const definition = ctx.tools.get(name)
      expect(definition).toBeDefined()
      const rendered = definition?.output.render({}, { status: 'ready' })
      expect(rendered?.[0]).toMatchObject({ type: 'text' })
    }
  })

  it('unskips a node through the edit tool', async () => {
    const { ctx } = await mount()
    await call(ctx, 'wf_edit', { node: 'a', action: 'skip' })
    expect((await call(ctx, 'wf_state')) as { skipped: string[] }).toMatchObject({ skipped: ['a'] })
    await call(ctx, 'wf_edit', { node: 'a', action: 'unskip' })
    expect((await call(ctx, 'wf_state')) as { skipped: string[] }).toMatchObject({ skipped: [] })
  })
})

describe('asking a person', () => {
  it.each(['native', 'pending'])('consumes a %s candidate revision once through the real Ledger bridge', async (channel) => {
    const workdir = scratch()
    const reads = ['.factory/strategy/current.md', '.factory/strategy/user-intent.md']
    mkdirSync(join(workdir, '.factory/strategy'), { recursive: true })
    for (const name of reads) writeFileSync(join(workdir, name), 'Original human requirements.')
    writeFileSync(join(workdir, 'facts.json'), JSON.stringify({ create_fidelity: true }))
    const python = resolve('.venv/bin/python')
    const commandEnv = { PYTHONPATH: resolve('refactory') }
    const feedback = 'Preserve every original requirement; test both invocations.'
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(UserQuestionService)
    const questions: string[] = []
    root.on('user-questions/request', (request: { questions: Array<{ id: string; question: string }>; signal?: AbortSignal }) => {
      const question = request.questions[0]!
      questions.push(question.question)
      if (channel === 'pending') {
        const queued = spawnSync(python, ['-c',
          'from pathlib import Path; from refactory_dsh.steering import Ledger; import json; '
          + 'l=Ledger(Path.cwd()); r=json.loads(' + JSON.stringify(JSON.stringify(reads)) + '); '
          + 'l.note("correction", ' + JSON.stringify(feedback) + '); '
          + 'l.verdict("gate_strategy","reloop",review=l.review("gate_strategy",r))'],
        { cwd: workdir, env: { ...process.env, ...commandEnv }, encoding: 'utf8', timeout: 10_000 })
        expect(queued.status, queued.stderr).toBe(0)
        return new Promise<never>((_resolve, reject) => request.signal?.addEventListener('abort',
          () => reject(new Error('pending question released after Ledger answer')), { once: true }))
      }
      return Promise.resolve({ answers: [{ id: question.id, selected: question.id === 'gate_strategy' ? ['reloop'] : [],
        custom: question.id === 'gate_strategy' ? '' : feedback }] })
    })
    await root.plugin(spinePlugin, config({ workdir, gateCommand: `'${python}' -m refactory_dsh.steering`, commandEnv,
      graph: { name: 'candidate-revision', start_node: 'gate_strategy', nodes: {
        gate_strategy: { id: 'gate_strategy', _type: 'GateNode', evaluator_type: 'user', reads,
          gate_prompt: 'INTENT FIDELITY CHECK: preserve the original asks and Graph Topology.', max_iterations: 3 },
        strategy: { id: 'strategy', _type: 'AgentNode', prompt_template: 'Revise from the human feedback.' },
        builder: { id: 'builder', _type: 'AgentNode', prompt_template: 'Build the approved strategy.' },
      }, edges: [{ source: 'gate_strategy', target: 'strategy', condition: 'reloop' },
        { source: 'strategy', target: 'gate_strategy' }, { source: 'gate_strategy', target: 'builder', condition: 'proceed' }] },
    }))
    await expect(call(root, 'wf_verdict', { gate: 'gate_strategy' }))
      .resolves.toMatchObject({ status: 'routed', verdict: 'reloop', answered_by: 'user' })
    expect(questions[0]).toContain('INTENT FIDELITY CHECK')
    expect(questions).toHaveLength(channel === 'native' ? 2 : 1)
    expect((readFileSync(join(workdir, reads[1]!), 'utf8').match(/Preserve every original requirement/g) ?? [])).toHaveLength(1)
    const receipts = readdirSync(join(workdir, '.refactory/create-fidelity-gates'))
      .map(id => JSON.parse(readFileSync(join(workdir, '.refactory/create-fidelity-gates', id, 'verdict.json'), 'utf8')))
    expect(receipts).toHaveLength(1)
    expect(receipts[0]).toMatchObject({ owner: 'human', gate_id: 'gate_strategy', verdict: 'reloop', status: 'answered' })
    expect(await call(root, 'wf_state')).toMatchObject({ cursor: ['strategy'], gate_verdicts: { gate_strategy: 'reloop' } })
  })

  it.each(['pause', 'stop'])('applies %s while a gate question is pending without recording a verdict', async (action) => {
    const workdir = scratch()
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(UserQuestionService)
    let questionSignal: AbortSignal | undefined
    const asked = Promise.withResolvers<void>()
    root.on('user-questions/request', (request: { signal?: AbortSignal }) => {
      questionSignal = request.signal
      asked.resolve()
      return new Promise<never>((_resolve, reject) => request.signal?.addEventListener('abort',
        () => reject(new Error('pending native question released')), { once: true }))
    })
    await root.plugin(spinePlugin, config({ workdir, graph: {
      name: 'pending-control', start_node: 'gate', nodes: {
        gate: { id: 'gate', _type: 'GateNode', evaluator_type: 'user', reads: [] },
        after: { id: 'after', _type: 'AgentNode', prompt_template: 'must remain undispatched' },
      }, edges: [{ source: 'gate', target: 'after', condition: 'proceed' }],
    } }))
    const pending = call(root, 'wf_verdict', { gate: 'gate' })
    await asked.promise
    expect(questionSignal?.aborted).toBe(false)
    const directory = join(workdir, '.refactory/graph-controls')
    mkdirSync(directory, { recursive: true })
    writeFileSync(join(directory, '1.json'), JSON.stringify({ action, run_id: workdir, invocation: null }))
    const result = await pending
    expect(result).toMatchObject({ status: action === 'pause' ? 'paused' : 'halted', interrupted_by: 'graph_control' })
    expect(result).not.toHaveProperty('verdict')
    expect(questionSignal?.aborted).toBe(true)
    const state = await call(root, 'wf_state')
    expect(state).toMatchObject({ cursor: ['gate'], completed: [], gate_verdicts: {} })
    expect(JSON.parse(readFileSync(join(workdir, '.refactory/spine-status.json'), 'utf8')))
      .toMatchObject({ status: action === 'pause' ? 'paused' : 'halted' })
    expect(JSON.parse(readFileSync(join(workdir, '.refactory/graph-control-receipt.json'), 'utf8')))
      .toMatchObject({ action, status: 'applied' })
  })

  it('refuses an answer that carries no verdict', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(UserQuestionService)
    root.on('user-questions/request', (request: { questions: Array<{ id: string }> }) =>
      Promise.resolve({ answers: request.questions.map(question => ({ id: question.id, selected: [] })) }))
    await root.plugin(spinePlugin, config({
      graph: {
        name: 't',
        start_node: 'gate',
        nodes: { gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] } },
        edges: [{ source: 'gate', target: 'gate', condition: 'proceed' }],
      },
    }))
    await expect(call(root, 'wf_verdict', { gate: 'gate' }))
      .rejects.toThrow(/answered without a verdict/)
  })

  it('forwards the calling agent and its abort signal to the answerer', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(UserQuestionService)
    let seen: { agent?: unknown; signal?: unknown } = {}
    root.on('user-questions/request', (request: { questions: Array<{ id: string }>; agent?: unknown; signal?: unknown }) => {
      seen = { agent: request.agent, signal: request.signal }
      return Promise.resolve({ answers: request.questions.map(question => ({ id: question.id, selected: ['proceed'] })) })
    })
    await root.plugin(spinePlugin, config({
      graph: {
        name: 't',
        start_node: 'gate',
        nodes: { gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] } },
        edges: [{ source: 'gate', target: 'gate', condition: 'proceed' }],
      },
    }))
    const controller = new AbortController()
    const agent = { id: 'agent-1' } as unknown as Agent
    // The question service accepts only the registry's exact live root agent.
    root.provide('agents', { get: (id: string) => (id === 'agent-1' ? agent : undefined), roots: () => [agent] })
    const definition = root.tools.get('wf_verdict')
    await definition?.execute({ gate: 'gate' }, { agent, signal: controller.signal } as never)
    expect(seen.agent).toBe(agent)
    expect(seen.signal).toBeInstanceOf(AbortSignal)
    expect((seen.signal as AbortSignal).aborted).toBe(true) // the resolved question no longer owns a polling loop
  })

  it('asks a gate that lists no outcomes as a free-text question', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(UserQuestionService)
    let asked: { options?: unknown } = {}
    root.on('user-questions/request', (request: { questions: Array<{ id: string; options?: unknown }> }) => {
      asked = request.questions[0] ?? {}
      return Promise.resolve({ answers: [{ id: 'gate', selected: [], custom: 'proceed' }] })
    })
    await root.plugin(spinePlugin, config({
      graph: {
        name: 't',
        start_node: 'gate',
        nodes: { gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] } },
        edges: [{ source: 'gate', target: 'gate' }],
      },
    }))
    await expect(call(root, 'wf_verdict', { gate: 'gate' }))
      .resolves.toMatchObject({ status: 'routed', verdict: 'proceed', answered_by: 'user' })
    expect(asked.options).toBeUndefined()
  })

  it('records a refusal when the answerer itself fails, whatever it throws', async () => {
    const workdir = scratch()
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(UserQuestionService)
    // A listener may reject with anything, so the plugin must report it rather than
    // crash; a rejection that is not an Error is the case under test.
    const notAnError = { toString: () => 'no answerer here' }
    root.on('user-questions/request', () => Promise.reject(notAnError))
    await root.plugin(spinePlugin, config({
      workdir,
      graph: {
        name: 't',
        start_node: 'gate',
        nodes: { gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] } },
        edges: [{ source: 'gate', target: 'gate' }],
      },
    }))
    await expect(call(root, 'wf_verdict', { gate: 'gate' }))
      .rejects.toThrow(/could not be asked: no answerer here/)
    expect(JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8')))
      .toMatchObject({ status: 'refused' })
  })
})

describe('the CEO deny guard', () => {
  function agentWith(ctx: Context | undefined, origin = 'root'): Agent {
    return { id: 'agent-1', session: { header: { origin } }, ctx } as unknown as Agent
  }

  /**
   * An agent context inside the composition. The agent runtime mints its context
   * from the agent's own composition, so `tools` resolves there; a bare scope
   * context does not carry that injection, and the proxy supplies the service the
   * guard reads from the agent's side.
   */
  function agentCtxIn(root: Context, parent: object): Context {
    const scoped = createScope(root, {}, { parent }).ctx
    return new Proxy(scoped, {
      get: (target, prop, receiver): unknown => (prop === 'tools' ? root.tools : Reflect.get(target, prop, receiver)),
    })
  }

  /** A root with the injected services, a scoped composition, and the plugin mounted. */
  async function composed(deny: string[]): Promise<{
    root: Context
    composition: Context
    compositionKey: object
    restrict: MockInstance<typeof ToolRuntime.prototype.restrict>
  }> {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    const compositionKey = {}
    const composition = createScope(root, compositionKey).ctx
    await composition.plugin(spinePlugin, config({ ceoDeny: deny }))
    // The tool registry is one instance; a scoped context resolves the same one.
    const restrict = vi.spyOn(root.tools, 'restrict')
    return { root, composition, compositionKey, restrict }
  }

  it('denies a named tool the CEO can actually reach, and disposes with the agent', async () => {
    const { root, composition, compositionKey, restrict } = await composed(['wf_next', 'ghost-tool'])
    const disposed: number[] = []
    restrict.mockImplementation(() => () => { disposed.push(1) })
    const agent = agentWith(agentCtxIn(root, compositionKey))

    composition.emit('agent/created', { agent, source: 'startup' })
    expect(restrict).toHaveBeenCalledWith({ deny: ['wf_next'] })

    composition.emit('agent/disposed', { agent })
    expect(disposed).toHaveLength(1)
  })

  it('leaves a subagent alone', async () => {
    const { root, composition, compositionKey, restrict } = await composed(['wf_next'])
    const agentCtx = createScope(root, {}, { parent: compositionKey }).ctx
    composition.emit('agent/created', { agent: agentWith(agentCtx, 'subagent'), source: 'startup' })
    expect(restrict).not.toHaveBeenCalled()
  })

  it('leaves an agent outside the composition alone', async () => {
    const { root, composition, restrict } = await composed(['wf_next'])
    const elsewhere = createScope(root, {}).ctx
    composition.emit('agent/created', { agent: agentWith(elsewhere), source: 'startup' })
    expect(restrict).not.toHaveBeenCalled()
  })

  it('skips the guard when the composition is unscoped', async () => {
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    await root.plugin(spinePlugin, config({ ceoDeny: ['wf_next'] }))
    const restrict = vi.spyOn(root.tools, 'restrict')
    root.emit('agent/created', { agent: agentWith(root), source: 'startup' })
    expect(restrict).not.toHaveBeenCalled()
  })

  it('leaves an unscoped agent alone', async () => {
    const { root, composition, restrict } = await composed(['wf_next'])
    composition.emit('agent/created', { agent: agentWith(root), source: 'startup' })
    expect(restrict).not.toHaveBeenCalled()
  })

  it('denies nothing when the list names tools the CEO cannot reach', async () => {
    const { root, composition, compositionKey, restrict } = await composed(['not-a-tool', 'pwsh'])
    const agentCtx = createScope(root, {}, { parent: compositionKey }).ctx
    composition.emit('agent/created', { agent: agentWith(agentCtx), source: 'startup' })
    expect(restrict).not.toHaveBeenCalled()
  })

  it('disposes an agent the guard never restricted without failing', async () => {
    const { root, composition } = await composed(['wf_next'])
    expect(() => { composition.emit('agent/disposed', { agent: agentWith(root) }) }).not.toThrow()
  })
})
