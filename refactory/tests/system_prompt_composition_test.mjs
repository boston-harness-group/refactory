/** The actual DSH overlay, system-prompt service, and pinned Claude SDK request boundary. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { readFileSync, realpathSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { pathToFileURL } from 'node:url'
import vm from 'node:vm'

const [root, requestPath] = process.argv.slice(2)
const request = JSON.parse(readFileSync(requestPath, 'utf8'))
for (const key of ['REFACTORY_SYSTEM_PROMPT', 'REFACTORY_MODEL_EFFORT']) {
  if (request.env[key]) process.env[key] = request.env[key]
  else delete process.env[key]
}
const project = createRequire(join(root, 'package.json'))
const installed = createRequire(project.resolve('@deepseek-ai/dsh/package.json'))
const boot = await import(pathToFileURL(installed.resolve('@deepseek-ai/dsh-app-boot')).href)
const base = join(dirname(installed.resolve('@deepseek-ai/dsh-base/package.json')), 'cordis.patch.yml')
const headless = join(dirname(installed.resolve('@deepseek-ai/dsh-headless/package.json')), 'cordis.patch.yml')
const patch = JSON.parse(readFileSync(request.patch, 'utf8'))
const entries = boot.composeEntries([boot.loadOverlayPatches('test-base', base),
  boot.loadOverlayPatches('test-headless', headless), patch])
const systemRow = entries.find(row => row.id === 'refactory-system-prompt')
const systemPromptRequire = createRequire(installed.resolve('@deepseek-ai/dsh-system-prompt/package.json'))
const { Context } = systemPromptRequire('@deepseek-ai/cordis')
const SystemPrompt = (await import(pathToFileURL(systemPromptRequire.resolve('@deepseek-ai/dsh-system-prompt')).href)).default
const composition = new Context()
await composition.plugin(SystemPrompt)
if (systemRow) {
  const systemPlugin = await import(pathToFileURL(systemRow.name).href)
  systemPlugin.apply(composition)
}
const assembly = await composition.systemPrompt.assemble()
const section = assembly.sections.find(item => item.name === 'refactory-node-system')

const row = entries.find(item => item.id === 'refactory-external-worker')
assert.ok(row, 'DSH overlay has no external worker provider')
row.config = { ...row.config, env: {
  REFACTORY_SYSTEM_PROMPT: section?.text || '',
  REFACTORY_MODEL_EFFORT: request.env.REFACTORY_MODEL_EFFORT,
} }
const context = vm.createContext({ process, Buffer, AbortController, AbortSignal, setTimeout, clearTimeout,
  setImmediate, clearImmediate, console })
const resolve = createRequire(realpathSync(row.name))
let sdkInput, queryCalls = 0, nativeSpawns = 0, registered
const module = new vm.SourceTextModule(readFileSync(row.name, 'utf8'), {
  context, identifier: row.name, initializeImportMeta(meta) { meta.url = pathToFileURL(row.name).href },
})
await module.link(async specifier => {
  const exports = specifier === '@anthropic-ai/claude-agent-sdk'
    ? { query(input) { queryCalls++; sdkInput = input; throw new Error('scripted SDK boundary') } }
    : await import(specifier.startsWith('node:') ? specifier : pathToFileURL(resolve.resolve(specifier)).href)
  const names = Object.keys(exports)
  return new vm.SyntheticModule(names, function () {
    for (const name of names) this.setExport(name, exports[name])
  }, { context })
})
await module.evaluate()
const native = module.namespace
native.apply({
  subagents: { registerProvider(value) { registered = value } },
  logger: { warn() {} }, emit() {},
  subprocess: { spawn() { nativeSpawns++; throw new Error('native process forbidden') } },
}, native.Config(row.config))
await assert.rejects(registered.start({
  parent: { session: { id: 'system-prompt-test', header: { cwd: request.cwd } } },
  prompt: [{ type: 'text', text: request.task }],
  signal: new AbortController().signal,
}), /query-start/)
assert.equal(queryCalls, 1)
assert.equal(nativeSpawns, 0)
assert.equal(sdkInput.prompt, 'History: user asked to build')
assert.equal(sdkInput.options.systemPrompt.type, 'preset')
assert.equal(sdkInput.options.systemPrompt.preset, 'claude_code')
assert.equal(sdkInput.options.systemPrompt.append, 'You are a Python expert')
assert.equal(sdkInput.options.effort, 'medium')
console.log(JSON.stringify({ prompt: sdkInput.prompt, systemPrompt: sdkInput.options.systemPrompt,
  effort: sdkInput.options.effort, sections: assembly.sections.map(item => item.name) }))
