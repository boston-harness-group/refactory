/**
 * Minimal spine driver for the data failure equivalence test.
 *
 * Usage: node --import tsx/esm data-drain.mts <graph.json> <workdir> <python-bin> <timeoutMs>
 *
 * Mounts the spine plugin with a spawn-based shell, drives wf_next until the
 * run finishes, and exits.  The workdir must already contain declarations.json
 * (written by `lowering.write_declarations`).
 *
 * With DRAIN_RESUME=1 it first asks the spine to go on from the run's saved
 * checkpoint (as `graph-control resume` does), acknowledging a node whose
 * process was killed mid-command: how the pass-isolation tests resume a run
 * stopped in the middle of an item.
 */
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as spinePlugin from '../../src/index.ts'

const [graphFile, workdir, pythonBin, timeoutArg] = process.argv.slice(2)
if (!graphFile || !workdir) {
  process.stderr.write('usage: data-drain.mts <graph.json> <workdir> <python-bin> <timeoutMs>\n')
  process.exit(2)
}
const commandTimeoutMs = Number(timeoutArg ?? 10_000)
const graph = JSON.parse(readFileSync(graphFile!, 'utf8'))

function runCommand(spec: { command: string; workdir?: string; env?: Record<string, string>; timeoutMs: number }) {
  return new Promise<{
    exitCode: number | null; signal: string | null; timedOut: boolean; aborted: boolean; timeoutMs: number
    stdout: { text: string; truncated: boolean }; stderr: { text: string; truncated: boolean }
  }>(done => {
    const child = spawn('bash', ['--norc', '--noprofile', '-c', spec.command], {
      cwd: spec.workdir, env: { ...process.env, ...spec.env }, stdio: ['ignore', 'pipe', 'pipe'],
    })
    const out: Buffer[] = []
    const err: Buffer[] = []
    child.stdout.on('data', (chunk: Buffer) => out.push(chunk))
    child.stderr.on('data', (chunk: Buffer) => err.push(chunk))
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL') }, spec.timeoutMs)
    child.on('close', (code, signal) => {
      clearTimeout(timer)
      done({
        exitCode: code, signal, timedOut, aborted: false, timeoutMs: spec.timeoutMs,
        stdout: { text: Buffer.concat(out).toString('utf8'), truncated: false },
        stderr: { text: Buffer.concat(err).toString('utf8'), truncated: false },
      })
    })
  })
}

if (process.env.DRAIN_RESUME === '1') {
  const checkpoint = join(workdir!, '.refactory', 'spine-checkpoint.json')
  if (!existsSync(checkpoint)) {
    process.stderr.write('DRAIN_RESUME: the run has no checkpoint\n')
    process.exit(2)
  }
  writeFileSync(join(workdir!, '.refactory', 'resume-request.json'), JSON.stringify({
    checkpoint_sha256: createHash('sha256').update(readFileSync(checkpoint)).digest('hex'),
    acknowledge_executing: true,
  }))
}

const ctx = new Context()
await ctx.plugin(SystemPrompt)
await ctx.plugin(ToolRuntime)
ctx.provide('shell', {
  resolve: (spec: { command: string; workdir?: string; env?: Record<string, string>; timeoutMs?: number }) => ({
    command: spec.command,
    workdir: spec.workdir ?? workdir,
    timeoutMs: spec.timeoutMs ?? commandTimeoutMs,
    onExpiry: 'kill',
    stdoutMaxBytes: 10_000_000,
    env: spec.env,
  }),
  execute: async (spec: { command: string; workdir?: string; env?: Record<string, string>; timeoutMs: number }) => ({
    result: () => runCommand(spec),
  }),
})

await ctx.plugin(spinePlugin, {
  graph,
  toolPrefix: 'wf',
  maxIterations: 500,
  executeCommands: true,
  workdir,
  commandEnv: { REFACTORY_WORKDIR: workdir! },
  dataCommand: `${pythonBin ?? 'python'} -m refactory_dsh.items`,
  commandTimeoutMs,
  commandSandboxMode: 'danger-full-access',
  interactiveOpen: false,
  openCommand: '',
  openArgs: [],
  steerCommand: '',
  steerArgs: [],
  ceoDeny: [],
})

async function call(name: string, input: unknown = {}) {
  const tool = ctx.tools.get(name)
  if (!tool) throw new Error(`tool ${name} not registered`)
  return tool.execute(input, {} as never)
}

let status = 'unknown'
for (let step = 0; step < 1000; step += 1) {
  const next = await call('wf_next') as { status?: string; nodes?: Array<{ id: string; _type: string }> }
  status = next.status ?? 'unknown'
  if (status !== 'ready') break
  // For fully deterministic graphs the drain handles everything.
  const agents = (next.nodes ?? []).filter(n => n._type === 'AgentNode' || n._type === 'LLMNode')
  if (agents.length > 0) {
    process.stderr.write(`unexpected agent node: ${agents[0]!.id}\n`)
    process.exit(1)
  }
  for (const node of (next.nodes ?? [])) {
    await call('wf_done', { node: node.id })
  }
}

console.log(JSON.stringify({ status }))
process.exit(status === 'done' ? 0 : 1)
