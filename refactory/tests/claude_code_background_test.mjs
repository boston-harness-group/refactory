/** The patched Claude Code adapter waits for background tasks before a step finishes.
 * Runs consumeClaudeQuery against scripted SDK streams; no product or model starts. */
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import vm from 'node:vm'
import assert from 'node:assert/strict'

const root = process.argv[2]
const source = readFileSync(resolve(root, 'node_modules/@deepseek-ai/dsh-subagent-claude-code/lib/index.js'), 'utf8')
// From the stream helpers consumeClaudeQuery uses to its end.
const start = source.indexOf('const GIVEN_UP')
const scope = {
  setTimeout, clearTimeout,
  ClaudeCodeFailure: class extends Error { constructor(facts) { super(facts?.detail); this.facts = facts } },
  successfulResult: message => {
    if (message.is_error) throw new Error(`error result: ${message.result}`)
    return message.result
  },
  resultFailureFacts: () => ({ stage: 'query-run', category: 'invalid-result', detail: 'the answer was empty' }),
}
vm.createContext(scope)
vm.runInContext(source.slice(start, source.indexOf('\n/**', source.indexOf('async function consumeClaudeQuery('))), scope)

const tasks = (...ids) => ({ type: 'system', subtype: 'background_tasks_changed',
  tasks: ids.map(id => typeof id === 'string' ? { task_id: id, task_type: 'local_agent', description: id } : id) })
const result = text => ({ type: 'result', subtype: 'success', is_error: false, result: text })

/** Run a script; `released()` inside it says whether the input was released by then. */
async function run(script, sent = undefined, abortQuery = undefined, askAgain = undefined) {
  let released = false
  let release
  const releasedNow = new Promise(resolve => { release = resolve })
  const metadata = []
  const stream = script(() => released, releasedNow)
  const outcome = await scope.consumeClaudeQuery(stream, undefined, undefined, data => metadata.push(data),
    sent, () => { released = true; release() }, abortQuery, askAgain).then(value => ({ value }), error => ({ error }))
  return { ...outcome, released, metadata }
}

// (a) A result while a task runs keeps the input open, also across a repeated init; the
// result after the set empties is the answer and releases the input.
const seen = []
const waited = await run(async function* (released) {
  yield tasks('bg-1')
  yield result('launched')
  seen.push(released())
  yield { type: 'system', subtype: 'init', model: 'm', session_id: 's' }
  yield result('still waiting')
  seen.push(released())
  yield tasks()
  yield result('FINAL: bg-token')
  seen.push(released())
})
assert.deepEqual(seen, [false, false, true])
assert.equal(waited.value.output[0].text, 'FINAL: bg-token')
assert.equal(waited.metadata.at(-1).background_tasks_waited, 1)

// (b) No background tasks: the first result releases the input, as before.
const plain = []
const once = await run(async function* (released) {
  yield result('done')
  plain.push(released())
})
assert.deepEqual(plain, [true])
assert.equal(once.value.output[0].text, 'done')
assert.ok(!once.metadata.some(data => 'background_tasks_waited' in data))

// (c) An error result, or a stream that throws, still releases the input.
const failed = await run(async function* () {
  yield tasks('bg-1')
  yield { type: 'result', subtype: 'success', is_error: true, result: 'boom' }
})
assert.match(String(failed.error), /error result: boom/)
assert.equal(failed.released, true)
const thrown = await run(async function* () {
  yield tasks('bg-1')
  throw new Error('stream broke')
})
assert.match(String(thrown.error), /stream broke/)
assert.equal(thrown.released, true)

// (d) Ambient tasks never hold a step.
const ambient = []
await run(async function* (released) {
  yield tasks({ task_id: 'monitor', task_type: 'local_bash', description: 'watch', ambient: true })
  yield result('done')
  ambient.push(released())
})
assert.deepEqual(ambient, [true])
// (e) A task that stays silent past the quiet bound releases the input with the last
// answer (Claude Code then ends and stops it), and is recorded as abandoned.
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
const quiet = await run(async function* (released, releasedNow) {
  yield tasks('server')
  yield result('server started')
  await releasedNow  // the CLI ends once its input closes
}, { backgroundQuietS: 0.1 })
assert.equal(quiet.value.output[0].text, 'server started')
assert.equal(quiet.released, true)
assert.equal(quiet.metadata.at(-1).background_tasks_abandoned, 1)

// (f) Any message inside the bound restarts it: progress every 60 ms keeps a 100 ms
// bound from firing, and the later answer is the step's.
const kept = []
const progressing = await run(async function* (released) {
  yield tasks('bg-1')
  yield result('launched')
  for (let i = 0; i < 4; i += 1) {
    await sleep(60)
    yield { type: 'system', subtype: 'task_progress', task_id: 'bg-1' }
  }
  kept.push(released())
  yield tasks()
  yield result('FINAL')
}, { backgroundQuietS: 0.1 })
assert.deepEqual(kept, [false])
assert.equal(progressing.value.output[0].text, 'FINAL')
assert.ok(!progressing.metadata.some(data => 'background_tasks_abandoned' in data))
// (g) A CLI that stays alive and silent after that release is stopped after a short
// bound: the step finishes with its last answer, and records that it stopped the stream.
let aborted = 0
const silent = await run(async function* () {
  yield tasks('server')
  yield result('server started')
  await new Promise(() => {})  // never ends, even once its input is closed
}, { backgroundQuietS: 0.05 }, () => { aborted += 1 })
assert.equal(silent.value.output[0].text, 'server started')
assert.equal(aborted, 1)
assert.equal(silent.metadata.find(data => 'background_stream_stopped' in data).background_stream_stopped, true)
assert.equal(silent.metadata.find(data => 'background_tasks_abandoned' in data).background_tasks_abandoned, 1)
assert.ok(!quiet.metadata.some(data => 'background_stream_stopped' in data))
// (h) A CLI still streaming after that release is not stopped: every message restarts
// the post-release bound too, so its last turn's result is the answer.
let abortedWhileStreaming = 0
const streaming = await run(async function* (released, releasedNow) {
  yield tasks('bg-1')
  yield result('early')
  await releasedNow
  for (let i = 0; i < 8; i += 1) {
    await sleep(40)
    yield { type: 'system', subtype: 'task_progress', task_id: 'bg-1' }
  }
  yield tasks()
  yield result('FINAL')
}, { backgroundQuietS: 0.1 }, () => { abortedWhileStreaming += 1 })
assert.equal(streaming.value.output[0].text, 'FINAL')
assert.equal(abortedWhileStreaming, 0)
assert.ok(!streaming.metadata.some(data => 'background_stream_stopped' in data))

// (i) The run that lost its answer: real results while tasks run, then the last turn
// after the final report ends with no text. The step asks once in the same session, and
// that turn's answer is the step's.
const asked = []
const answered = []
const reasked = await run(async function* (released) {
  yield tasks('bg-1')
  yield result('waiting for the researcher')
  yield { type: 'system', subtype: 'init', model: 'm', session_id: 's' }
  yield tasks()
  yield result('')
  answered.push(released())
  yield result('FINAL: plan written')
}, undefined, undefined, text => { asked.push(text); return true })
assert.equal(asked.length, 1)
assert.match(asked[0], /ended without a reply/)
assert.deepEqual(answered, [false])
assert.equal(reasked.value.output[0].text, 'FINAL: plan written')
assert.equal(reasked.released, true)
assert.ok(reasked.metadata.some(data => data.empty_answer_reasked === true))
assert.ok(!reasked.metadata.some(data => 'answer_from_earlier_turn' in data))
// (j) Still no text when asked, and the only text was written while a task ran: that is
// an interim note, not the answer, so the step fails and the failure carries it.
const interimOnly = await run(async function* () {
  yield tasks('bg-1')
  yield result('launching the researchers')
  yield tasks()
  yield result('')
  yield result('  ')
}, undefined, undefined, () => true)
assert.equal(interimOnly.error.facts.detail, 'the answer was empty, also when asked once more for it; '
  + 'its last text, written while background tasks still ran, was: launching the researchers')
// (k) No text in any turn, even when asked: the empty-answer failure, saying it asked.
let askedOnce = 0
const none = await run(async function* () {
  yield result('')
  yield result('')
}, undefined, undefined, () => { askedOnce += 1; return true })
assert.equal(askedOnce, 1)
assert.equal(none.error.facts.detail, 'the answer was empty, also when asked once more for it')
// (l) A settled answer (given with no task running) is kept when a later turn -- Claude
// Code continuing on its own after the input closed -- ends with no text.
const settledKept = await run(async function* () {
  yield result('FULL ANSWER')
  yield tasks('late')
  yield tasks()
  yield result('')
}, undefined, undefined, () => false)
assert.equal(settledKept.value.output[0].text, 'FULL ANSWER')
assert.ok(settledKept.metadata.some(data => data.answer_from_earlier_turn === true))
const closedInterim = await run(async function* () {
  yield tasks('bg-1')
  yield result('interim note')
  yield tasks()
  yield result('')
}, undefined, undefined, () => false)
assert.match(closedInterim.error.facts.detail, /^the answer was empty; its last text, written while background tasks still ran, was: interim note$/)
const closedNone = await run(async function* () { yield result('') }, undefined, undefined, () => false)
assert.equal(closedNone.error.facts.detail, 'the answer was empty')
// (m) A reply to the re-ask that only points back never replaces a settled answer.
const pointer = await run(async function* () {
  yield result('FULL ANSWER')
  yield result('')
  yield result('My final answer is above.')
}, undefined, undefined, () => true)
assert.equal(pointer.value.output[0].text, 'FULL ANSWER')
// (n) An empty result while a task still runs is no reason to ask: Claude Code carries on
// as the task reports back, and the later answer is the step's.
let askedWhileLive = 0
const live = await run(async function* () {
  yield tasks('bg-1')
  yield result('')
  yield tasks()
  yield result('FINAL')
}, undefined, undefined, () => { askedWhileLive += 1; return true })
assert.equal(askedWhileLive, 0)
assert.equal(live.value.output[0].text, 'FINAL')
// (o) A CLI that never answers the re-ask is bounded like a silent task: the input is
// released after the quiet bound and the step fails, not held until the node timeout.
const unanswered = await run(async function* (released, releasedNow) {
  yield tasks('bg-1')
  yield result('interim note')
  yield tasks()
  yield result('')
  await releasedNow
}, { backgroundQuietS: 0.1 }, undefined, () => true)
assert.equal(unanswered.released, true)
assert.match(unanswered.error.facts.detail, /also when asked once more for it; its last text.*interim note$/)
// With no task ever running, only the bound armed by the re-ask can release the input.
const unansweredAlone = await Promise.race([
  run(async function* (released, releasedNow) {
    yield result('')
    await releasedNow
  }, { backgroundQuietS: 0.1 }, undefined, () => true),
  sleep(2000).then(() => 'held'),
])
assert.notEqual(unansweredAlone, 'held', 'an unanswered re-ask held the step past its quiet bound')
assert.equal(unansweredAlone.error.facts.detail, 'the answer was empty, also when asked once more for it')
console.log('claude code background-task waits passed')
