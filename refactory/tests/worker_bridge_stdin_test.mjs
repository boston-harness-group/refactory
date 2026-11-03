/** Run the real worker bridge with fake services; record the prompt it hands the worker.
 *
 * argv: <bridge.mjs> <startup.json> <out.json>. startup.json is the headlessStartup
 * service (its `task` may be absent). The prompt text passed to subagents.start is
 * written to out.json, with the exit code the bridge asked for.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [bridgePath, startupPath, outPath] = process.argv.slice(2)
const prompts = []
const settled = Promise.withResolvers()
const services = {
  loader: { await: async () => {} },
  headlessStartup: JSON.parse(readFileSync(startupPath, 'utf8')),
  agents: { create: async () => ({ agent: { session: { id: 'parent' } }, dispose: async () => {} }) },
  subagents: {
    start: async (name, request) => {
      prompts.push(request.prompt.map(block => block.text).join(''))
      return {
        result: Promise.resolve({ output: [{ type: 'text', text: 'ok' }], stopReason: 'completed' }),
        dispose: async () => {},
      }
    },
  },
  appExit: code => settled.resolve(code),
}
const bridge = await import(pathToFileURL(bridgePath).href)
bridge.apply({ get: key => services[key], on() {} })
const code = await settled.promise
writeFileSync(outPath, JSON.stringify({ code, prompts }))
