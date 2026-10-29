/** Real DSH system-prompt composition through AgentLoop to a scripted LLM adapter. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { readFileSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { pathToFileURL } from 'node:url'

const [root, requestPath] = process.argv.slice(2)
const request = JSON.parse(readFileSync(requestPath, 'utf8'))
for (const key of ['REFACTORY_SYSTEM_PROMPT', 'REFACTORY_MODEL_EFFORT']) {
  if (request.env[key]) process.env[key] = request.env[key]
  else delete process.env[key]
}
const project = createRequire(join(root, 'package.json'))
const installed = createRequire(project.resolve('@deepseek-ai/dsh/package.json'))
const packagePath = (prefix, packageName) => {
  const folder = readdirSync(join(root, 'node_modules/.pnpm')).find(name => name.startsWith(prefix))
  assert.ok(folder, `missing installed package ${prefix}`)
  return join(root, 'node_modules/.pnpm', folder, 'node_modules', packageName)
}
const boot = await import(pathToFileURL(installed.resolve('@deepseek-ai/dsh-app-boot')).href)
const base = join(dirname(installed.resolve('@deepseek-ai/dsh-base/package.json')), 'cordis.patch.yml')
const headless = join(dirname(installed.resolve('@deepseek-ai/dsh-headless/package.json')), 'cordis.patch.yml')
const patch = JSON.parse(readFileSync(request.patch, 'utf8'))
const entries = boot.composeEntries([boot.loadOverlayPatches('test-base', base),
  boot.loadOverlayPatches('test-headless', headless), patch])
const policyRow = entries.find(row => row.id === 'refactory-no-tools')
assert.ok(policyRow, 'generated overlay omitted no-tools enforcement')

const systemPromptRequire = createRequire(installed.resolve('@deepseek-ai/dsh/package.json'))
const { Context } = systemPromptRequire('@deepseek-ai/cordis')
const testkit = await import(pathToFileURL(join(packagePath('@deepseek-ai+dsh-agent-loop-testkit@', '@deepseek-ai/dsh-agent-loop-testkit'), 'lib/index.js')).href)
const llmPath = packagePath('@deepseek-ai+dsh-llm@', '@deepseek-ai/dsh-llm')
const { LlmAdapter, createUserMessage } = await import(pathToFileURL(join(llmPath, 'lib/index.js')).href)
const ctx = new Context()
await testkit.mountAgentLoopTestDependencies(ctx)
const policyPlugin = await import(pathToFileURL(policyRow.name).href)
policyPlugin.apply(ctx)
let executions = 0
const extra = {
  name: 'arbitrary_custom_tool_not_in_a_known_list', description: 'Hostile test tool',
  parameters: { type: 'object', properties: {} },
  output: { schema: { type: 'string' }, render: value => [{ type: 'text', text: value }] },
  async execute() { executions++; return 'executed' },
}
ctx.tools.register(extra)
const errors = []
ctx.on('agent/error', ({ error }) => errors.push(String(error)))
if (request.scoped) ctx.on('agent/created', ({ agent }) => agent.ctx.tools.register({ ...extra, name: 'late_scoped_tool' }))
let requests = 0
let finalRequest
class ScriptedAdapter extends LlmAdapter {
  providerInfo(provider) { return { id: provider, name: 'Scripted provider' } }
  async listModels() { return [] }
  async resolveModel(provider, model) { return { provider, id: model, name: model } }
  async *stream(options) {
    requests++
    finalRequest = options
    if (request.hostile && requests === 1) {
      const block = { type: 'tool-call', id: 'hostile-call', name: extra.name, arguments: '{}' }
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
const handle = await harness.create('system-prompt-test', {
  provider: 'scripted', model: 'scripted-model',
}, { cwd: request.cwd })
try {
  handle.followup(createUserMessage({ content: [{ type: 'text', text: request.task }] }))
  await handle.whenIdle()
  if (request.scoped) {
    assert.equal(requests, 0, 'late scoped tool reached the provider')
    assert.ok(errors.some(error => error.includes('tool_policy=none')), JSON.stringify(errors))
  } else {
    assert.ok(finalRequest, `no model request completed: ${errors}`)
    assert.equal(finalRequest.tools?.length ?? 0, 0)
    const user = finalRequest.messages.find(message => message.role === 'user')
    assert.ok(JSON.stringify(user.content).includes(request.task))
    assert.equal(requests, request.hostile ? 2 : 1)
  }
  assert.equal(executions, 0, 'a forbidden tool body executed')
  console.log(JSON.stringify({ requests, executions, tools: finalRequest?.tools ?? [], errors }))
} finally {
  handle.cancel({ kind: "disposed" })
}
