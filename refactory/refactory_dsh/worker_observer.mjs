/** Nonsecret route/usage receipts from DSH's public lifecycle/request seam. */
import { readFileSync, writeFileSync, renameSync, appendFileSync, realpathSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { createRequire } from 'node:module'
export const name = 'refactory-worker-observer'
export const inject = ['agents', 'subagents']
export function apply(ctx) {
  const path = process.env.REFACTORY_WORKER_RECEIPT
  if (!path) return
  const initial = JSON.parse(readFileSync(path, 'utf8'))
  const sessions = new Map()
  let rootSessionId
  const streams = new Map()
  let incompleteNestedUsage = false
  const calls = []
  const nativeObservations = []
  let lastDelta = null
  function usage() {
    const known = calls.reduce((sum, call) => sum + call.tokens, 0)
    // Every journal record used to carry the whole call log, so each new record
    // repeated every earlier one and the journal grew quadratically. A count is
    // the fact a reader needs; the log itself stays in this observer's memory.
    return incompleteNestedUsage
      ? { status: 'unavailable', tokens: null, known_tokens: known, tokens_delta: lastDelta, calls: calls.length,
        incomplete_nested_usage: true, reason: 'A delegated runtime has incomplete nested token coverage.' }
      : { status: 'measured', tokens: known, tokens_delta: lastDelta, calls: calls.length }
  }
  function save(patch, state, summary) {
    const receipt = { ...JSON.parse(readFileSync(path, 'utf8')), ...patch }
    const temp = path + '.runtime.tmp'
    writeFileSync(temp, JSON.stringify(receipt, null, 2) + '\n')
    renameSync(temp, path)
    // Both places get the same bounded summary: what a reader needs is how many
    // requests were measured, not every one of them again.
    if (state) appendFileSync(join(dirname(dirname(path)), 'events.jsonl'), JSON.stringify({
      schema_version: 1, at: new Date().toISOString(), kind: 'worker',
      run_id: initial.run_id, node_id: initial.node_id, worker_id: initial.worker_id,
      state, summary, route: receipt.actual_route, usage: receipt.usage,
    }) + '\n')
  }
  const backend = initial.requested_route.worker_backend
  let dshVersion = null
  try { dshVersion = JSON.parse(readFileSync(join(process.env.REFACTORY_RUNTIME_ROOT, 'node_modules/@deepseek-ai/dsh/package.json'), 'utf8')).version } catch {}
  let productVersion = null
  if (backend !== 'dsh') {
    try {
      const adapter = join(process.env.REFACTORY_RUNTIME_ROOT, 'node_modules',
        backend === 'codex' ? '@deepseek-ai/dsh-subagent-codex' : '@deepseek-ai/dsh-subagent-claude-code', 'package.json')
      const resolve = createRequire(realpathSync(adapter))
      const manifest = backend === 'codex' ? resolve.resolve('@openai/codex/package.json')
        : join(dirname(resolve.resolve('@anthropic-ai/claude-agent-sdk')), 'package.json')
      productVersion = JSON.parse(readFileSync(manifest, 'utf8')).version
    } catch {}
  }
  const runtime = { backend: backend === 'dsh' ? 'dsh' : null, requested_backend: backend,
    adapter: '@deepseek-ai/dsh-subagent', adapter_version: dshVersion,
    version: backend === 'dsh' ? dshVersion : null, installed_product_version: productVersion,
    version_kind: backend === 'claude_code' ? 'agent-sdk' : 'runtime', source: 'installed-package' }
  save({ runtime })
  ctx.on('refactory/worker-metadata', (parentId, metadata) => {
    if (backend !== 'dsh') {
      if (metadata.actual_route?.worker_backend === backend) {
        runtime.backend = backend
        runtime.version = productVersion
        runtime.source = 'started-pinned-provider'
      }
      save({ ...metadata, runtime }, 'metadata', 'Native worker route or usage observed')
      return
    }
    // A native descendant's model/usage is not the DSH coordinator's route or
    // complete usage total. Retain the observation with its delegating parent.
    incompleteNestedUsage = true
    nativeObservations.push({ parent_session_id: parentId, ...metadata })
    save({ native_observations: nativeObservations, usage: usage() }, 'metadata', 'Delegated native usage is incomplete')
  })
  ctx.on('agent/request', async ({ agent }, next) => {
    if (backend !== 'dsh') throw new Error(`External ${backend} worker cannot dispatch a DSH model request; check the worker-driver composition`)
    const ledger = process.env.REFACTORY_BUDGET_LEDGER
    if (ledger && JSON.parse(readFileSync(ledger, 'utf8')).limits?.max_tokens != null) {
      // The shipped spawn/fork providers are in-process and observed here.
      // There is no public pre-start interception hook for arbitrary providers;
      // refuse incompatible compositions before allowing a model/tool turn.
      const unsupported = ctx.subagents.list().filter(name => !['spawn', 'fork'].includes(name))
      if (unsupported.length) throw new Error(`Token-limited DSH requires in-process spawn/fork providers; unmetered providers: ${unsupported.join(', ')}`)
    }
    let route = await next()
    if (backend === 'dsh' && initial.requested_route.effort) {
      route = { ...route, reasoningEffort: initial.requested_route.effort }
    }
    const actual = { worker_backend: 'dsh', provider: route.provider, model: route.model,
      effort: route.reasoningEffort ?? null, source: 'runtime-request' }
    rootSessionId ??= agent.session.id
    sessions.set(agent.session.id, actual)
    save({ ...(agent.session.id === rootSessionId ? { actual_route: actual, runtime_session_id: rootSessionId } : {}),
      sessions: Object.fromEntries(sessions) }, 'running',
      `DSH request: ${route.provider}/${route.model}`)
    return route
  })
  ctx.on('agent/assistant-stream', ({ agent, frame }) => {
    const key = `${agent.session.id}:${frame.attemptId ?? 'current'}`
    if (frame.type === 'start') streams.set(agent.session.id, { id: key, usage: undefined })
    if (frame.type === 'chunk' && frame.chunk.type === 'usage') {
      const current = streams.get(agent.session.id) ?? { id: key }
      current.usage = frame.chunk.usage
      streams.set(agent.session.id, current)
    }
    if (frame.type !== 'end') return
    const current = streams.get(agent.session.id)
    if (!current?.usage) {
      incompleteNestedUsage = true
      save({ usage: usage() }, 'usage', 'A completed request did not report token usage')
      streams.delete(agent.session.id)
      return
    }
    const measured = current.usage
    const total = measured.totalTokens ?? (measured.inputTokens + measured.outputTokens + (measured.cacheReadTokens ?? 0) + (measured.cacheWriteTokens ?? 0))
    if (!Number.isSafeInteger(total) || total < 0) {
      incompleteNestedUsage = true
      save({ usage: usage() }, 'usage', 'A completed request reported incomplete token usage')
      streams.delete(agent.session.id)
      return
    }
    lastDelta = total
    calls.push({ session_id: agent.session.id, tokens: total })
    save({ usage: usage() },
      'usage', `Measured ${total} tokens`)
    lastDelta = null
    streams.delete(agent.session.id)
  })
  ctx.on('subagent/start', info => {
    if (backend === 'dsh' && !info.local) incompleteNestedUsage = true
    save({ backend_session_id: info.id, ...(incompleteNestedUsage ? { usage: usage() } : {}) },
      'running', `Worker accepted by ${info.provider}`)
  })
  ctx.on('subagent/end', info => save({ stop_reason: info.stopReason }, info.stopReason, `Worker ${info.stopReason}`))
}
