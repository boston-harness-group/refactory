import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import assert from 'node:assert/strict'
import { apply } from '../refactory_dsh/worker_observer.mjs'

const root = mkdtempSync(join(tmpdir(), 'refactory-observer-'))
try {
  const directory = join(root, '.refactory', 'workers')
  mkdirSync(directory, { recursive: true })
  const path = join(directory, 'one.json')
  const deadlines = { dispatches: [{ attempt: 1, sampled_at: '2026-10-12T12:00:00+00:00',
    stage_remaining_s: 60, run_remaining_s: 40, effective_remaining_s: 40,
    idle_timer: { status: 'inactive', timeout_s: null } }] }
  writeFileSync(path, JSON.stringify({ requested_route: { worker_backend: 'dsh', effort: 'high' }, worker_id: 'one', run_id: root,
    usage: { status: 'unavailable', tokens: null }, effective_deadlines: deadlines }))
  process.env.REFACTORY_WORKER_RECEIPT = path
  const events = new Map()
  let providers = ['spawn', 'fork']
  apply({ on: (name, callback) => events.set(name, callback), subagents: { list: () => providers } })
  const agent = { session: { id: 'dsh-parent' } }
  const request = events.get('agent/request')
  const stream = events.get('agent/assistant-stream')
  const child = { session: { id: 'dsh-child' } }
  let childRoute
  const rootRoute = await request({ agent }, async () => {
    // Nested request middleware can resolve before the root's next() returns.
    childRoute = await request({ agent: child }, async () => ({ provider: 'child-p', model: 'child-model', reasoningEffort: 'low' }))
    return { provider: 'dsh-p', model: 'dsh-model', reasoningEffort: 'medium' }
  })
  assert.equal(rootRoute.reasoningEffort, 'high')
  assert.equal(childRoute.reasoningEffort, 'low', 'root effort must not replace a descendant override')
  const defaultChild = { session: { id: 'dsh-default-child' } }
  const defaultRoute = await request({ agent: defaultChild }, async () => ({ provider: 'child-p', model: 'child-model' }))
  assert.equal(defaultRoute.reasoningEffort, undefined, 'root effort must not fabricate a descendant setting')
  stream({ agent, frame: { type: 'start' } })
  stream({ agent, frame: { type: 'chunk', chunk: { type: 'usage', usage: { totalTokens: 11 } } } })
  stream({ agent, frame: { type: 'end' } })
  events.get('refactory/worker-metadata')('dsh-parent', { actual_route: { model: 'external-model' }, usage: { tokens: 90, status: 'measured' } })
  events.get('subagent/start')({ id: 'external', provider: 'codex', local: false })
  stream({ agent, frame: { type: 'start' } })
  stream({ agent, frame: { type: 'chunk', chunk: { type: 'usage', usage: { totalTokens: 13 } } } })
  stream({ agent, frame: { type: 'end' } })
  const receipt = JSON.parse(readFileSync(path))
  assert.deepEqual(receipt.effective_deadlines, deadlines, 'runtime observations must retain owner deadline samples')
  assert.equal(receipt.actual_route.model, 'dsh-model')
  assert.equal(receipt.actual_route.effort, 'high')
  assert.equal(receipt.sessions['dsh-child'].effort, 'low')
  assert.equal(receipt.sessions['dsh-default-child'].effort, null)
  assert.equal(receipt.usage.status, 'unavailable')
  assert.equal(receipt.usage.tokens, null)
  assert.equal(receipt.usage.known_tokens, 24)
  assert.equal(receipt.native_observations[0].actual_route.model, 'external-model')

  const ledger = join(root, 'budget.json')
  writeFileSync(ledger, JSON.stringify({ limits: { max_tokens: 1000 } }))
  process.env.REFACTORY_BUDGET_LEDGER = ledger
  providers = ['spawn', 'fork', 'external']
  let called = false
  await assert.rejects(request({ agent }, async () => { called = true }), /unmetered providers/)
  assert.equal(called, false)
  providers = ['spawn', 'fork']
  await request({ agent }, async () => ({ provider: 'dsh-p', model: 'dsh-model' }))
  // A miscomposed external worker must fail before entering DSH transport.
  writeFileSync(path, JSON.stringify({ requested_route: { worker_backend: 'codex' }, worker_id: 'external', run_id: root,
    usage: { status: 'unavailable', tokens: null } }))
  apply({ on: (name, callback) => events.set(name, callback), subagents: { list: () => [] } })
  let fallback = false
  await assert.rejects(events.get('agent/request')({ agent }, async () => { fallback = true }), /cannot dispatch a DSH model request/)
  assert.equal(fallback, false)
  assert.equal(JSON.parse(readFileSync(path)).runtime.backend, null)
  assert.equal(JSON.parse(readFileSync(path)).runtime.version, null)
  console.log('worker observer accounting and capability boundaries passed')
} finally {
  rmSync(root, { recursive: true, force: true })
}
