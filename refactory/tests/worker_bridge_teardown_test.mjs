/** Run the real worker bridge whose cleanup fails after the worker answered.
 *
 * argv: <bridge.mjs> <stop reason> <failing: none|worker|parent|both> <out.json>. Writes
 * the exit code the bridge asked for and what it wrote to stdout and stderr.
 */
import { writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [bridgePath, stopReason, failing, outPath] = process.argv.slice(2)
const settled = Promise.withResolvers()
const written = { stdout: '', stderr: '' }
const stdout = process.stdout.write.bind(process.stdout)
const stderr = process.stderr.write.bind(process.stderr)
process.stdout.write = chunk => { written.stdout += chunk; return true }
process.stderr.write = chunk => { written.stderr += chunk; return true }
const dispose = where => async () => {
  if (failing === where || failing === 'both') {
    throw new Error(`Product subagent failure (product: Claude Code; stage: teardown; category: unknown; exit code: 0) [${where}]`)
  }
}
const services = {
  loader: { await: async () => {} },
  headlessStartup: { provider: 'p', model: 'm', task: 'Solve.' },
  agents: { create: async () => ({ agent: { session: { id: 'parent' } }, dispose: dispose('parent') }) },
  subagents: {
    start: async () => ({
      result: Promise.resolve({ output: [{ type: 'text', text: 'the answer' }], stopReason, diagnostic: 'stopped' }),
      dispose: dispose('worker'),
    }),
  },
  appExit: code => settled.resolve(code),
}
const bridge = await import(pathToFileURL(bridgePath).href)
bridge.apply({ get: key => services[key], on() {} })
const code = await settled.promise
process.stdout.write = stdout
process.stderr.write = stderr
writeFileSync(outPath, JSON.stringify({ code, ...written }))
