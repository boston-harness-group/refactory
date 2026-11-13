/** Tests the pinned provider patches without launching a product or model. */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import vm from 'node:vm'
import assert from 'node:assert/strict'
const root = process.argv[2]
const codex = readFileSync(resolve(root, 'node_modules/@deepseek-ai/dsh-subagent-codex/lib/index.js'), 'utf8')
const start = codex.indexOf('var CodexAppServerWire = class')
const end = codex.indexOf('//#endregion', start)
const scope = { THREAD_PERMISSION_PARAMS: { never: {} },
  object: value => value, string: value => value, Promise,
}
vm.createContext(scope)
vm.runInContext(codex.slice(start, end), scope)
const wire = Object.create(scope.CodexAppServerWire.prototype)
wire.permissionMode = 'never'
wire.guarded = async value => value
wire.transport = { request: async () => ({ model: 'actual-codex', modelProvider: 'openai', reasoningEffort: 'high', thread: { id: 'thread-one', ephemeral: true } }) }
const observed = []
wire.onRefactoryMetadata = data => observed.push(data)
await wire.startThread('/work', new AbortController().signal)
assert.equal(observed[0].actual_route.model, 'actual-codex')
assert.equal(observed[0].product_session_id, 'thread-one')
wire.handleNotification('thread/tokenUsage/updated', { threadId: 'thread-one', tokenUsage: { total: { totalTokens: 123 } } })
assert.equal(observed.at(-1).usage.tokens, 123)
wire.handleNotification('model/rerouted', { threadId: 'thread-one', toModel: 'routed-model' })
assert.equal(observed.at(-1).actual_route.model, 'routed-model')

const claude = readFileSync(resolve(root, 'node_modules/@deepseek-ai/dsh-subagent-claude-code/lib/index.js'), 'utf8')
// From the stream helpers consumeClaudeQuery uses to its end.
const cStart = claude.indexOf('const GIVEN_UP')
const cEnd = claude.indexOf('\n/**', claude.indexOf('async function consumeClaudeQuery('))
const claudeScope = { successfulResult: message => message.result }
vm.createContext(claudeScope)
vm.runInContext(claude.slice(cStart, cEnd), claudeScope)
async function* query() {
  yield {type:'system',subtype:'init',model:'actual-claude',session_id:'claude-one'}
  yield {type:'result',result:'done',modelUsage:{'actual-claude':{inputTokens:2,outputTokens:3,cacheReadInputTokens:4,cacheCreationInputTokens:5}}}
}
const seen = []
const result = await claudeScope.consumeClaudeQuery(query(),undefined,undefined, data=>seen.push(data))
assert.equal(result.stopReason,'completed')
assert.equal(seen[0].actual_route.model,'actual-claude')
assert.equal(seen.at(-1).usage.tokens,14)
assert.equal(seen.at(-1).usage.status,'estimated')
assert.ok(seen.at(-1).usage.excludes.includes('permission'))
console.log('pinned provider metadata boundaries passed')
