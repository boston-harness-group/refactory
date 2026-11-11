/** One worker session through the real observer, as DSH reports a request's usage.
 *
 * argv: <worker_observer.mjs> <session id> <tokens>. The receipt is the one the model
 * client names in REFACTORY_WORKER_RECEIPT, so a client retry's sessions share it.
 * <tokens> "start" loads the observer and stops (a session that died before its first
 * request); "request" makes the request and reports no usage.
 */
import { pathToFileURL } from 'node:url'

const [observerPath, sessionId, tokens] = process.argv.slice(2)
const { apply } = await import(pathToFileURL(observerPath).href)
const events = new Map()
apply({ on: (name, callback) => events.set(name, callback), subagents: { list: () => ['spawn', 'fork'] } })
const agent = { session: { id: sessionId } }
if (tokens === 'start') process.exit(0)
await events.get('agent/request')({ agent }, async () => ({ provider: 'p', model: 'm' }))
if (tokens === 'request') process.exit(0)
const stream = events.get('agent/assistant-stream')
stream({ agent, frame: { type: 'start' } })
stream({ agent, frame: { type: 'chunk', chunk: { type: 'usage', usage: { totalTokens: Number(tokens) } } } })
stream({ agent, frame: { type: 'end' } })
