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
const systemRow = entries.find(row => row.id === 'refactory-system-prompt')
assert.ok(systemRow, 'DSH overlay omitted the system-prompt composition plugin')

const systemPromptRequire = createRequire(installed.resolve('@deepseek-ai/dsh/package.json'))
const { Context } = systemPromptRequire('@deepseek-ai/cordis')
const testkit = await import(pathToFileURL(join(packagePath('@deepseek-ai+dsh-agent-loop-testkit@', '@deepseek-ai/dsh-agent-loop-testkit'), 'lib/index.js')).href)
const llmPath = packagePath('@deepseek-ai+dsh-llm@', '@deepseek-ai/dsh-llm')
const { LlmAdapter, createUserMessage } = await import(pathToFileURL(join(llmPath, 'lib/index.js')).href)
const ctx = new Context()
await testkit.mountAgentLoopTestDependencies(ctx)
const systemPlugin = await import(pathToFileURL(systemRow.name).href)
systemPlugin.apply(ctx)

let finalRequest
class ScriptedAdapter extends LlmAdapter {
  providerInfo(provider) { return { id: provider, name: 'Scripted provider' } }
  async listModels() { return [] }
  async resolveModel(provider, model) { return { provider, id: model, name: model } }
  async *stream(options) {
    finalRequest = options
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
  assert.ok(finalRequest, 'no request reached the DSH LLM provider boundary')
  const system = finalRequest.messages.find(message => message.role === 'system')
  const user = finalRequest.messages.find(message => message.role === 'user')
  assert.ok(system, 'DSH final provider request has no system message')
  assert.ok(user, 'DSH final provider request has no user history')
  assert.match(JSON.stringify(system.content), /You are a Python expert/)
  assert.match(JSON.stringify(user.content), /History: user asked to build/)
  console.log(JSON.stringify({ provider: finalRequest.provider, model: finalRequest.model,
    messages: finalRequest.messages, sections: entries.map(entry => entry.id) }))
} finally {
  handle.cancel({ kind: "disposed" })
}
