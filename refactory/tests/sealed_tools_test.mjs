/** A sealed step through DSH's own pieces (`test_sealed_tools.py`).
 *
 * provider: DSH's local sandbox provider with refactory's runner as its runnerCommand
 *   wraps `bash -c` as the bash tool asks it to; the wrapped argv is run and reported.
 * plugin: the generated overlay composes onto DSH's base and headless profiles; its tools
 *   plugin, through a real AgentLoop and a scripted model, offers only the listed tools
 *   and never executes another.
 */
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { pathToFileURL } from 'node:url'

const [root, requestPath, mode] = process.argv.slice(2)
const request = JSON.parse(readFileSync(requestPath, 'utf8'))
const packagePath = (prefix, packageName) => {
  const folder = readdirSync(join(root, 'node_modules/.pnpm')).find(name => name.startsWith(prefix))
  assert.ok(folder, `missing installed package ${prefix}`)
  return join(root, 'node_modules/.pnpm', folder, 'node_modules', packageName)
}

if (mode === 'provider') {
  const lib = join(packagePath('@deepseek-ai+dsh-sandbox-local@', '@deepseek-ai/dsh-sandbox-local'), 'lib/index.js')
  const { Context } = createRequire(lib)('@deepseek-ai/cordis')
  const { LocalSandboxProvider } = await import(pathToFileURL(lib).href)
  const provider = new LocalSandboxProvider(new Context(), {
    runnerCommand: request.runner, runnerFailureSignatures: [request.failure], probeTimeoutMs: 5000,
  })
  const wrapped = await provider.confine(['bash', '-c', request.command],
    { mode: 'workspace-write', workspaceRoot: request.workspace })
  const done = spawnSync(wrapped.argv[0], wrapped.argv.slice(1), { cwd: request.workspace, encoding: 'utf8' })
  console.log(JSON.stringify({ argv: wrapped.argv, enforcement: wrapped.enforcement,
    status: done.status, stdout: done.stdout, stderr: done.stderr }))
} else {
  const project = createRequire(join(root, 'package.json'))
  const installed = createRequire(project.resolve('@deepseek-ai/dsh/package.json'))
  const boot = await import(pathToFileURL(installed.resolve('@deepseek-ai/dsh-app-boot')).href)
  const base = join(dirname(installed.resolve('@deepseek-ai/dsh-base/package.json')), 'cordis.patch.yml')
  const headless = join(dirname(installed.resolve('@deepseek-ai/dsh-headless/package.json')), 'cordis.patch.yml')
  const patch = JSON.parse(readFileSync(request.patch, 'utf8'))
  const entries = boot.composeEntries([boot.loadOverlayPatches('test-base', base),
    boot.loadOverlayPatches('test-headless', headless), patch])
  const sandbox = entries.find(row => row.id === 'sandbox')
  assert.ok(sandbox && String(sandbox.name).includes('dsh-sandbox-local'), 'the runner is not on DSH\'s provider')
  const toolsRow = entries.find(row => row.id === 'refactory-tools')
  assert.ok(toolsRow, 'generated overlay omitted the tools plugin')

  const { Context } = installed('@deepseek-ai/cordis')
  const testkit = await import(pathToFileURL(join(packagePath('@deepseek-ai+dsh-agent-loop-testkit@', '@deepseek-ai/dsh-agent-loop-testkit'), 'lib/index.js')).href)
  const { LlmAdapter, createUserMessage } = await import(pathToFileURL(join(packagePath('@deepseek-ai+dsh-llm@', '@deepseek-ai/dsh-llm'), 'lib/index.js')).href)
  const ctx = new Context()
  await testkit.mountAgentLoopTestDependencies(ctx)
  const counts = { bash: 0, hostile: 0 }
  const tool = (name) => ({
    name, description: `test ${name}`, parameters: { type: 'object', properties: {} },
    output: { schema: { type: 'string' }, render: value => [{ type: 'text', text: value }] },
    async execute() { counts[name === 'bash' ? 'bash' : 'hostile']++; return 'executed' },
  })
  ctx.tools.register(tool('bash'))
  ctx.tools.register(tool('web_fetch'))
  const plugin = await import(pathToFileURL(toolsRow.name).href)
  plugin.apply(ctx, toolsRow.config)
  let requests = 0
  let offered
  class ScriptedAdapter extends LlmAdapter {
    providerInfo(provider) { return { id: provider, name: 'Scripted provider' } }
    async listModels() { return [] }
    async resolveModel(provider, model) { return { provider, id: model, name: model } }
    async *stream(options) {
      requests++
      offered ??= (options.tools ?? []).map(t => t.name).sort()
      const call = { 1: 'web_fetch', 2: 'bash' }[requests]
      if (call) {
        const block = { type: 'tool-call', id: `call-${requests}`, name: call, arguments: '{}' }
        yield { type: 'block-start', index: 0, blockType: 'tool-call' }
        yield { type: 'block-end', index: 0, block }
        yield { type: 'finish', reason: { kind: 'stop' } }
        return
      }
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text: 'done' }
      yield { type: 'block-end', index: 0, block: { type: 'text', text: 'done' } }
      yield { type: 'finish', reason: { kind: 'stop' } }
    }
  }
  ctx.llm.registerAdapter(['scripted'], new ScriptedAdapter())
  const harness = await testkit.mountAgentLoopTestHarness(ctx)
  const handle = await harness.create('sealed-tools-test', { provider: 'scripted', model: 'scripted-model' },
    { cwd: request.cwd })
  try {
    handle.followup(createUserMessage({ content: [{ type: 'text', text: 'go' }] }))
    await handle.whenIdle()
    console.log(JSON.stringify({ sandbox: sandbox.config, offered, requests,
      hostile_executions: counts.hostile, bash_executions: counts.bash }))
  } finally {
    handle.cancel({ kind: 'disposed' })
  }
}
