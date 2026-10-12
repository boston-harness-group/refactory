import { existsSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { pathToFileURL } from 'node:url'
import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it } from 'vitest'
import type { ShellExecution, ShellExecRequest, ShellExecSpec, ShellExecutor, ShellRunResult } from '@deepseek-ai/dsh-shell'
import { commandOf, isDeterministic, NODE_ENV_KEY, runDeterministic, verdictOf } from '../src/execute.ts'
import type { CommandFailure, CommandOutcome, CommandRunnerConfig, NodeManifest } from '../src/execute.ts'
import type { NodeView } from '../src/spine.ts'

/**
 * A shell executor that really runs the command, so these tests exercise the
 * process boundary rather than a stub's opinion of it.
 */
const localShell = {
  resolve(request: ShellExecRequest): ShellExecSpec {
    return {
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: request.timeoutMs ?? 10_000,
      onExpiry: request.onExpiry ?? 'kill',
      stdoutMaxBytes: 1_000_000,
      env: request.env,
      sandboxPolicy: undefined,
    }
  },
  async execute(spec: ShellExecSpec): Promise<ShellExecution> {
    const started = spawnSync('/bin/sh', ['-c', spec.command], {
      cwd: spec.workdir,
      encoding: 'utf8',
      timeout: spec.timeoutMs,
      env: { ...process.env, ...spec.env },
    })
    const result: ShellRunResult = {
      exitCode: started.status,
      signal: started.signal,
      timedOut: started.error !== undefined && 'code' in started.error && started.error.code === 'ETIMEDOUT',
      aborted: false,
      timeoutMs: spec.timeoutMs,
      stdout: { text: started.stdout ?? '', truncated: false },
      stderr: { text: started.stderr ?? '', truncated: false },
    }
    return { result: async () => result } as unknown as ShellExecution
  },
} as unknown as ShellExecutor

/** The failure of a call the test expected to fail. */
function failure(outcome: CommandOutcome): CommandFailure {
  if (outcome.ok) throw new Error('expected the node to fail')
  return outcome
}

/** The success of a call the test expected to complete. */
function completed(outcome: CommandOutcome): Extract<CommandOutcome, { ok: true }> {
  if (!outcome.ok) throw new Error(`expected the node to complete: ${outcome.error}`)
  return outcome
}

const config: CommandRunnerConfig = { env: {}, timeoutMs: 10_000 }

function fn(id: string, command?: string): NodeView {
  return command === undefined
    ? { id, _type: 'FnNode', reads: [], writes: [] }
    : { id, _type: 'FnNode', command, reads: ['in.txt'], writes: ['out.txt'] }
}

function gate(id: string, evaluatorCommand?: string): NodeView {
  const view: NodeView = { id, _type: 'GateNode', evaluator_type: 'fn', reads: [] }
  if (evaluatorCommand !== undefined) view.evaluator_command = evaluatorCommand
  return view
}

describe('which nodes the plugin runs itself', () => {
  it('executes the candidate CEO adapter while retaining logical agent ownership', async () => {
    const view: NodeView = { id: 'gate_build', _type: 'GateNode', evaluator_type: 'agent',
      evaluator_command: 'python -m refactory_dsh.run refactory_dsh.create_fidelity_gates:evaluate_native',
      gate_prompt: 'Original CEO criteria' }
    expect(commandOf(view)).toBe(view.evaluator_command)
    expect(view.evaluator_type).toBe('agent')
    expect(commandOf({ ...view, evaluator_command: 'echo proceed' })).toBeUndefined()
    expect(commandOf({ ...view, evaluator_type: 'user' })).toBeUndefined()
  })
  it('claims an FnNode with a command and an fn gate with an evaluator', () => {
    expect(isDeterministic(fn('a', 'true'))).toBe(true)
    expect(isDeterministic(gate('g', 'echo proceed'))).toBe(true)
  })

  it('leaves nodes a model owns to the driver', () => {
    expect(isDeterministic(fn('a'))).toBe(false)
    expect(isDeterministic({ id: 'l', _type: 'LLMNode' })).toBe(false)
    expect(isDeterministic({ id: 'a', _type: 'AgentNode', prompt_template: 'x' })).toBe(false)
    expect(isDeterministic({ id: 'g', _type: 'GateNode', evaluator_type: 'user' })).toBe(false)
    expect(isDeterministic(gate('g'))).toBe(false)
  })

  it('reports the command it would run', () => {
    expect(commandOf(fn('a', 'echo hi'))).toBe('echo hi')
    expect(commandOf(gate('g', 'echo proceed'))).toBe('echo proceed')
    expect(commandOf(fn('a'))).toBeUndefined()
  })
})

describe('running a deterministic node', () => {
  it('carries a declared stage wall through the real pinned DSH shell clamp and subprocess', async () => {
    const store = resolve('node_modules/.pnpm')
    const installed = async (name: string): Promise<{ default: new (context: Context, config?: Record<string, unknown>) => object }> => {
      const directory = readdirSync(store).find(entry => entry.startsWith(`@deepseek-ai+${name}@`))
      if (directory === undefined) throw new Error(`pinned ${name} runtime is unavailable`)
      return import(pathToFileURL(resolve(store, directory, `node_modules/@deepseek-ai/${name}/lib/index.js`)).href)
    }
    const subprocess = await installed('dsh-subprocess-local')
    const bash = await installed('dsh-bash-local')
    const ctx = new Context()
    await ctx.plugin(subprocess.default)
    await ctx.plugin(bash.default, { timeoutMs: 20, maxTimeoutMs: 20, graceMs: 100 })
    try {
      const shell = ctx.get('shell') as ShellExecutor
      expect(shell.resolve({ command: 'true', timeoutMs: 1_000 }).timeoutMs).toBe(20)
      const outcome = await runDeterministic(shell, { ...gate('original_stage', 'sleep 0.05; printf proceed'), command_timeout_s: 1 }, config)
      expect(completed(outcome).verdict).toBe('proceed')
    } finally { await ctx.fiber.dispose() }
  })
  it('runs an FnNode command and reports success', async () => {
    const outcome = await runDeterministic(localShell, fn('a', 'printf done'), config)
    expect(outcome.ok).toBe(true)
    expect(outcome.exitCode).toBe(0)
    expect(outcome.stdout).toBe('done')
  })

  it('reports a nonzero exit instead of throwing', async () => {
    const outcome = await runDeterministic(localShell, fn('a', 'echo boom >&2; exit 3'), config)
    expect(outcome.ok).toBe(false)
    expect(outcome.exitCode).toBe(3)
    expect(failure(outcome).error).toContain('exited with code 3')
    expect(outcome.stderr).toBe('boom')
  })

  it('refuses a node it was asked to run that has no command', async () => {
    const outcome = await runDeterministic(localShell, fn('a'), config)
    expect(outcome.ok).toBe(false)
    expect(failure(outcome).error).toBe('a has no command')
  })

  it('reports a timeout and an abort as failures, not as steps that happened', async () => {
    const stub = (result: Partial<ShellRunResult>): ShellExecutor => ({
      resolve: (request: ShellExecRequest): ShellExecSpec => ({
        command: request.command,
        workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000,
        onExpiry: request.onExpiry ?? 'kill',
        stdoutMaxBytes: 1_000,
        env: request.env,
        sandboxPolicy: undefined,
      }),
      execute: async () => ({
        result: async () => ({
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 250,
          stdout: { text: '', truncated: false }, stderr: { text: '', truncated: false },
          ...result,
        }),
      }),
    } as unknown as ShellExecutor)

    const timedOut = await runDeterministic(stub({ timedOut: true }), fn('a', 'true'), config)
    expect(timedOut.ok).toBe(false)
    expect(failure(timedOut).error).toBe('a timed out after 250ms')

    const aborted = await runDeterministic(stub({ aborted: true }), fn('a', 'true'), config)
    expect(aborted.ok).toBe(false)
    expect(failure(aborted).error).toBe('a was aborted')
  })

  it('runs in the configured work directory', async () => {
    const outcome = await runDeterministic(localShell, fn('a', 'pwd'), { ...config, workdir: '/tmp' })
    expect(outcome.ok).toBe(true)
    expect(outcome.stdout.endsWith('/tmp')).toBe(true)
  })

  it('passes the configured environment through', async () => {
    const outcome = await runDeterministic(localShell, fn('a', 'printf "$SRF_TASK"'), {
      ...config,
      env: { SRF_TASK: 'circle_packing' },
    })
    expect(outcome.stdout).toBe('circle_packing')
  })

  it('hands the node its own declaration, so the command knows what it reads and writes', async () => {
    const outcome = await runDeterministic(localShell, fn('build_prompt_0', `printf '%s' "$${NODE_ENV_KEY}"`), config)
    expect(outcome.ok).toBe(true)
    const manifest = JSON.parse(outcome.stdout) as NodeManifest
    expect(manifest).toEqual({
      id: 'build_prompt_0',
      type: 'FnNode',
      command: 'printf \'%s\' "$DSH_WORKFLOW_NODE"',
      reads: ['in.txt'],
      writes: ['out.txt'],
    })
  })

  it('declares empty reads and writes for a node that names none', async () => {
    const bare: NodeView = { id: 'bare', _type: 'FnNode', command: `printf '%s' "$${NODE_ENV_KEY}"` }
    const outcome = await runDeterministic(localShell, bare, config)
    const manifest = JSON.parse(outcome.stdout) as NodeManifest
    expect(manifest.reads).toEqual([])
    expect(manifest.writes).toEqual([])
  })

  it('carries the sandbox policy into the command spec when one is configured', async () => {
    let seen: ShellExecSpec | undefined
    const capturing = {
      resolve(request: ShellExecRequest): ShellExecSpec {
        seen = {
          command: request.command,
          workdir: request.workdir ?? process.cwd(),
          timeoutMs: request.timeoutMs ?? 10_000,
          onExpiry: request.onExpiry ?? 'kill',
          stdoutMaxBytes: 1_000,
          env: request.env,
          sandboxPolicy: request.sandboxPolicy,
        }
        return seen
      },
      execute: async () => ({
        result: async () => ({
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text: 'ok', truncated: false }, stderr: { text: '', truncated: false },
        }),
      }),
    } as unknown as ShellExecutor
    const policy = { mode: 'workspace-write' as const, workspaceRoot: '/w' }
    await runDeterministic(capturing, fn('a', 'true'), { ...config, sandboxPolicy: policy })
    expect(seen?.sandboxPolicy).toEqual(policy)
  })

  it('fails the node when its command produces no verdict', async () => {
    const outcome = await runDeterministic(localShell, gate('g', 'true'), config)
    expect(outcome.ok).toBe(false)
    expect(failure(outcome).error).toContain('printed no verdict')
  })
})

describe('reading a gate verdict', () => {
  it('takes the last non-empty stdout line, lowercased', () => {
    expect(verdictOf('noise\nPROCEED\n')).toBe('proceed')
    expect(verdictOf('  RELOOP  \n\n')).toBe('reloop')
    expect(verdictOf('DRAFT')).toBe('draft')
  })

  it('is undefined for empty output', () => {
    expect(verdictOf('')).toBeUndefined()
    expect(verdictOf('\n \n')).toBeUndefined()
  })

  it('routes a gate by its evaluator output', async () => {
    const outcome = await runDeterministic(localShell, gate('action_gate', 'echo DRAFT'), config)
    expect(outcome.ok).toBe(true)
    expect(completed(outcome).verdict).toBe('draft')
  })

  it('ignores evaluator log lines before the verdict', async () => {
    const outcome = await runDeterministic(localShell, gate('g', 'echo "log line"; echo RELOOP'), config)
    expect(completed(outcome).verdict).toBe('reloop')
  })
})


describe('shared resource boundaries', () => {
  it('uses serialized Design phases to finish QA but refuses repair from validation reserve', async () => {
    const root = mkdtempSync(join(tmpdir(), 'spine-repair-resources-'))
    try {
      const prepared = spawnSync(resolve('.venv/bin/python'), ['-c',
        'import json,sys; from pathlib import Path; from refactory_dsh.budgets import RunBudget,ResourceLimits; from refactory_dsh.fast import DesignFastMode; from refactory_dsh.lowering import to_workflow; b=RunBudget.open(Path(sys.argv[1]),ResourceLimits(max_tokens=100,validation_reserve_tokens=20)); b.record_usage("prior",85); print(json.dumps(to_workflow(DesignFastMode().state_dict()).to_dict()["nodes"]))', root], { encoding: 'utf8' })
      expect(prepared.status, prepared.stderr).toBe(0)
      const nodes = JSON.parse(prepared.stdout) as Record<string, NodeView>
      const bounded = { ...config, workdir: root, env: { PATH: `${resolve('.venv/bin')}:${process.env.PATH}`, REFACTORY_BUDGET_LEDGER: join(root, '.refactory/resource-usage.json') } }
      const phaseNode = (id: string, command: string): NodeView => ({ ...fn(id, command), ...(nodes[id]?.resource_role === undefined ? {} : { resource_role: nodes[id].resource_role }) })
      for (const id of ['focused_check', 'focused_qa', 'gate_qa', 'report']) {
        expect(completed(await runDeterministic(localShell, phaseNode(id, `echo ${id}`), bounded)).stdout).toBe(id)
      }
      await expect(runDeterministic(localShell, phaseNode('emit', 'touch repair-dispatched'), bounded)).rejects.toThrow(/reserve retained/)
      expect(existsSync(join(root, 'repair-dispatched'))).toBe(false)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })

  it('retains final validation reserve, then lets validation and reporting finish within it', async () => {
    const root = mkdtempSync(join(tmpdir(), 'spine-resources-'))
    try {
      const ledger = join(root, '.refactory/resource-usage.json')
      const prepared = spawnSync(resolve('.venv/bin/python'), ['-c',
        'from pathlib import Path; from refactory_dsh.budgets import RunBudget,ResourceLimits; b=RunBudget.open(Path(__import__("sys").argv[1]),ResourceLimits(max_tokens=100,validation_reserve_tokens=20)); b.record_usage("prior",85)', root], { encoding: 'utf8' })
      expect(prepared.status, prepared.stderr).toBe(0)
      const bounded = { ...config, workdir: root, env: { PATH: `${resolve('.venv/bin')}:${process.env.PATH}`, REFACTORY_BUDGET_LEDGER: ledger } }
      await expect(runDeterministic(localShell, fn('work', 'echo forbidden'), bounded)).rejects.toThrow(/reserve retained/)
      expect(completed(await runDeterministic(localShell, { ...fn('validate', 'echo validated'), resource_role: 'final_validation' }, bounded)).stdout).toBe('validated')
      expect(completed(await runDeterministic(localShell, fn('report', 'echo reported'), bounded)).stdout).toBe('reported')
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})
