// The worker observer touches the session's activity file on every streamed frame and
// native worker message (`model.activity_path`), at most once a second; the receipt
// itself is written only when a request starts and when a response ends.
import { mkdtempSync, mkdirSync, statSync, writeFileSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import assert from 'node:assert/strict'
import { setTimeout as sleep } from 'node:timers/promises'
import { apply } from '../refactory_dsh/worker_observer.mjs'

const root = mkdtempSync(join(tmpdir(), 'refactory-activity-'))
try {
  const directory = join(root, '.refactory', 'workers')
  mkdirSync(directory, { recursive: true })
  const path = join(directory, 'worker-one.json')
  const activity = join(directory, 'worker-one.activity')
  writeFileSync(path, JSON.stringify({ requested_route: { worker_backend: 'dsh' }, worker_id: 'one', run_id: root,
    usage: { status: 'unavailable', tokens: null } }))
  process.env.REFACTORY_WORKER_RECEIPT = path
  const events = new Map()
  apply({ on: (name, callback) => events.set(name, callback), subagents: { list: () => ['spawn', 'fork'] } })
  const agent = { session: { id: 'root' } }
  const stream = events.get('agent/assistant-stream')
  await events.get('agent/request')({ agent }, async () => ({ provider: 'p', model: 'm' }))
  stream({ agent, frame: { type: 'start' } })
  const receiptAt = statSync(path).mtimeMs
  assert.equal(existsSync(activity), true, 'the first frame creates the activity file')
  const first = statSync(activity).mtimeMs

  // A long thinking turn: deltas only, no receipt write, for over a second.
  await sleep(1100)
  stream({ agent, frame: { type: 'chunk', chunk: { type: 'thinking-delta', text: 'hmm' } } })
  const thinking = statSync(activity).mtimeMs
  assert.ok(thinking > first, 'a thinking delta is activity')
  assert.equal(statSync(path).mtimeMs, receiptAt, 'a delta does not rewrite the receipt')

  // A native (claude_code) worker's message, relayed by the adapter.
  await sleep(1100)
  events.get('refactory/worker-activity')('root')
  assert.ok(statSync(activity).mtimeMs > thinking, 'a native worker message is activity')

  // Throttled: a burst within a second touches the file once.
  const settled = statSync(activity).mtimeMs
  for (let i = 0; i < 50; i += 1) stream({ agent, frame: { type: 'chunk', chunk: { type: 'text-delta', text: 'x' } } })
  assert.equal(statSync(activity).mtimeMs, settled)
  console.log('worker observer activity passed')
} finally {
  rmSync(root, { recursive: true, force: true })
}
