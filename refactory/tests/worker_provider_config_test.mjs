/** Compose the real pinned headless profile; a declared provider route must land in it. */
import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import assert from 'node:assert/strict'
const [root, directory] = process.argv.slice(2)
const project = createRequire(join(root, 'package.json'))
const installed = createRequire(project.resolve('@deepseek-ai/dsh/package.json'))
const load = async name => import(pathToFileURL(installed.resolve(name)).href)
const { composeEntries, loadOverlayPatches } = await load('@deepseek-ai/dsh-app-boot')
const base = join(installed.resolve('@deepseek-ai/dsh-base/package.json'), '../cordis.patch.yml')
const headless = join(installed.resolve('@deepseek-ai/dsh-headless/package.json'), '../cordis.patch.yml')
const gateway = {
  apiKeyEnv: 'LITELLM_API_KEY', api: 'anthropic-messages',
  baseURL: 'https://gateway.example/v1',
  models: [{ id: 'gateway-model', name: 'gateway-model' }],
}
for (const name of ['with-provider.json', 'without-provider.json']) {
  const selected = JSON.parse(readFileSync(join(directory, name)))
  const entries = composeEntries([
    loadOverlayPatches('provider-config-base', base),
    loadOverlayPatches('provider-config-test', headless), selected,
  ])
  const adapter = entries.find(row => row.id === 'llm-pi-ai' && !row.disabled)
  assert.ok(adapter, 'the dormant base llm-pi-ai row must survive composition')
  if (name === 'with-provider.json') {
    assert.deepEqual(adapter.config.providers, { gateway })
    // The row the worker patch targets is the same row the nested runtime mounts:
    // mounting the real adapter with the composed config must accept the route.
    const { apply, Config } = await import(pathToFileURL(installed.resolve('@deepseek-ai/dsh-llm-pi-ai/package.json')).href
      .replace(/package\.json$/, 'lib/index.js'))
    const registered = []
    const profiles = Config(adapter.config)
    const services = {
      llm: {
        registerAdapter: value => registered.push(value),
        registerConfigurableProviders: () => ({ replace() {} }),
        registerModelDiscovery: () => ({ dispose() {} }),
      },
      settings: { configure: async () => {} },
      authorization: { registerFlow: () => ({ dispose() {} }) },
    }
    apply({
      llm: services.llm,
      get: key => services[key],
      // The authorization seam is scoped out of compositions without a sign-in
      // surface (headless among them); the stub never provides one.
      inject() {},
      on() {}, effect() {}, logger: { warn() {} }, emit() {},
      fiber: { entry: { options: { id: 'llm-pi-ai' } } },
    }, profiles)
    assert.ok(registered.length >= 1, 'a configured provider route registers the adapter')
  } else {
    assert.equal(adapter.config, undefined)
  }
}
console.log('actual DSH composition carries declared worker provider profiles; zero model calls')
