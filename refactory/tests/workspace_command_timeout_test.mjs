/** Real pinned shell/tool, real subprocess; no model calls. */
import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import assert from 'node:assert/strict'
const [root, patch] = process.argv.slice(2)
const project = createRequire(join(root, 'package.json'))
const installed = createRequire(project.resolve('@deepseek-ai/dsh/package.json'))
const base = createRequire(installed.resolve('@deepseek-ai/dsh-base/package.json'))
const load = async name => import(pathToFileURL(base.resolve(name)).href)
const { composeEntries, loadOverlayPatches } = await import(pathToFileURL(installed.resolve('@deepseek-ai/dsh-app-boot')).href)
const entries = composeEntries([
  loadOverlayPatches('command-timeout-test', base.resolve('@deepseek-ai/dsh-base/cordis.patch.yml')),
  loadOverlayPatches('command-timeout-test', installed.resolve('@deepseek-ai/dsh-headless/cordis.patch.yml')),
  JSON.parse(readFileSync(patch)),
])
const executorConfig = entries.find(row => row.id === 'bash-sandbox').config
const toolConfig = entries.find(row => row.id === 'tool-bash').config
assert.deepEqual(executorConfig, { timeoutMs: 600_000, maxTimeoutMs: 600_000 })
assert.deepEqual(toolConfig, { enableRunInBackground: false, promoteOnTimeout: false })
const { Context } = await load('@deepseek-ai/cordis')
const { default: Subprocess } = await load('@deepseek-ai/dsh-subprocess-local')
const { default: Shell } = await load('@deepseek-ai/dsh-bash-sandbox')
const tool = await load('@deepseek-ai/dsh-tool-bash')
const ctx = new Context()
ctx.provide('sandboxPolicy', { defaultMode: 'danger-full-access', resolve: () => ({mode: 'danger-full-access'}) })
ctx.provide('sandbox', {})
await ctx.plugin(Subprocess)
await ctx.plugin(Shell, executorConfig)
try {
  const shell = ctx.get('shell')
  assert.equal(shell.resolve({command: 'true', timeoutMs: 9e9}).timeoutMs, 600_000)
  assert.equal(shell.resolve({command: 'true'}).onExpiry, 'kill')
  let bash
  tool.apply({
    shell, shellEnv: { collect: () => ({}) },
    get: name => ctx.get(name),
    tools: { register: value => { bash = value } },
    systemPrompt: { section() {}, getSectionOrder: () => 0 },
  }, tool.Config(toolConfig))
  assert.ok(!('run_in_background' in bash.parameters.properties))
  const exec = { signal: new AbortController().signal }
  const call = (command, timeoutMs) => bash.execute({ command, description: 'Test workspace command deadline', timeoutMs }, exec)
  const result = await call('printf started; sleep 60', 5000)
  assert.equal(result.kind, 'foreground')
  assert.equal(result.timedOut, true)
  assert.match(result.stdout.text, /started/)
  const rendered = bash.output.render({}, result).map(part => part.text).join('\n')
  assert.match(rendered, /timed out/i)
  const next = await call('printf recovered', 5000)
  assert.equal(next.exitCode, 0)
  assert.equal(next.stdout.text, 'recovered')
  assert.equal(next.timedOut, false)
  await assert.rejects(() => bash.execute({command: 'sleep 60', description: 'Try background bypass', run_in_background: true}, exec), /disabled/)
} finally {
  await ctx.fiber.dispose()
}
console.log('hung shell command killed; timeout reported; next command succeeded')
