/**
 * Spine driver for the both-runtimes record equivalence test.
 *
 * Usage: node --import tsx/esm data-drain-agent.mts <graph.json> <workdir> <python-bin> <timeoutMs> <responses.json>
 *
 * Like data-drain.mts but handles agent nodes: reads scripted responses from
 * <responses.json> (a map from node-id to response text) and calls wf_done
 * with the scripted response.  The agent delivery bridge (agentCommand)
 * records the exchange in the sidecar, just as the real spine path does.
 */
import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as spinePlugin from '../../src/index.ts'

const [graphFile, workdir, pythonBin, timeoutArg, responsesFile] = process.argv.slice(2)
if (!graphFile || !workdir) {
  process.stderr.write('usage: data-drain-agent.mts <graph.json> <workdir> <python-bin> <timeoutMs> [responses.json]\n')
  process.exit(2)
}
const commandTimeoutMs = Number(timeoutArg ?? 10_000)
const graph = JSON.parse(readFileSync(graphFile!, 'utf8'))
const responses: Record<string, string> = responsesFile
  ? JSON.parse(readFileSync(responsesFile, 'utf8'))
  : {}

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
  agentCommand: `${pythonBin ?? 'python'} -m refactory_dsh.agent_delivery`,
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
  const next = await call('wf_next') as { status?: string; nodes?: Array<{ id: string; _type: string }>; error?: string }
  status = next.status ?? 'unknown'
  if (status === 'done') break
  if (status === 'failed') {
    // A node failure inside a data-node pass is expected for error items.
    // The drain continues so the remaining items can be processed.
    continue
  }
  if (status !== 'ready') break

  for (const node of (next.nodes ?? [])) {
    if (node._type === 'AgentNode' || node._type === 'LLMNode') {
      // Provide the scripted response for agent nodes.
      const response = responses[node.id] ?? `scripted response for ${node.id}`
      await call('wf_done', { node: node.id, output: response })
    } else {
      await call('wf_done', { node: node.id })
    }
  }
}

console.log(JSON.stringify({ status }))
process.exit(status === 'done' ? 0 : 1)
