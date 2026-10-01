/** The one explicit Remote contribution needed by the panel's launch stream. */
import type { TypertRemoteContribution } from '@deepseek-ai/dsh-typert-protocol'
import type { RefactorySingleEvent, RefactorySingleRunRequest, RefactoryWatchFrame } from '@deepseek-ai/dsh-refactory-console/types'

function record(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error('Expected a run stream object')
  return value as Record<string, unknown>
}
function strings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(item => typeof item === 'string')
}
function textFields(value: Record<string, unknown>, fields: string[]): boolean {
  return fields.every(field => typeof value[field] === 'string')
}
function request(value: unknown): RefactorySingleRunRequest {
  const entry = record(value)
  if (typeof entry.mode !== 'string' || (entry.knobs !== undefined && !strings(entry.knobs))) {
    throw new Error('Invalid Run once request')
  }
  return entry as unknown as RefactorySingleRunRequest
}

/**
 * One strict codec for this hand-written contribution. It carries both the
 * `schema.parse` face of the pinned harness and the `create()` factory that
 * `e459e32637` introduced, so the same descriptor mounts under either harness.
 *
 * Exit condition: drop the `schema` face only once **both** the harness this
 * repository pins and the harness serving the profile are `0.1.6-alpha.2` or
 * later — `e459e32637` shipped in `0.1.6-alpha.2`, so "when it is released" is
 * already past. #276 (the pin bump) plus `boston-harness-group/dsh-tauri#25`
 * (the serving harness) are what reach that state; until both land, the bridge
 * is load-bearing.
 * @param typeSymbol - generated type symbol used for diagnostics.
 * @param parse - boundary validator for the wire value.
 * @returns the strict codec accepted by both the pinned and current harness.
 */
function strictCodec(typeSymbol: string, parse: (value: unknown) => unknown) {
  return { mode: 'strict' as const, typeSymbol, schema: { parse }, create: () => ({ parse }) }
}

/** Validate every delivered frame, including the complete ordinary result. */
export function singleEvent(value: unknown): RefactorySingleEvent {
  const entry = record(value)
  if (entry.kind === 'launch') {
    const launch = record(entry.launch)
    if (!textFields(launch, ['mode', 'root']) || !['starting', 'prepared'].includes(String(launch.phase))
      || (launch.phase === 'prepared' && typeof launch.invocation !== 'string')) throw new Error('Invalid run launch identity')
  } else if (entry.kind === 'result') {
    const run = record(entry.run)
    if (!textFields(run, ['run_id', 'root', 'state']) || (run.mode !== null && typeof run.mode !== 'string')
      || typeof run.finished !== 'boolean' || (run.best_score !== null && typeof run.best_score !== 'number')
      || !Object.values(record(run.facts)).every(item => typeof item === 'string')) throw new Error('Invalid final run result')
    if (run.authoring !== null) {
      const authoring = record(run.authoring)
      if (!textFields(authoring, ['spine', 'error']) || !strings(authoring.nodes_due)) throw new Error('Invalid graph progress')
    }
    if (run.item_results != null) {
      const counts = record(run.item_results)
      if (!['count', 'passed', 'failed', 'unclassified'].every(key => typeof counts[key] === 'number')) throw new Error('Invalid item results')
    }
  } else throw new Error('Unknown run stream frame')
  return entry as unknown as RefactorySingleEvent
}

/**
 * The lean assembly has no generated namespace for this repository's service.
 * Mount the stream through the published Gateway so browser WebSocket framing,
 * cancellation and errors retain their existing owners. Unary calls still use
 * Connection RPC. The descriptor and these boundary validators live together.
 */
export const runStreamContribution: TypertRemoteContribution = {
  package: '@deepseek-ai/dsh-refactory-console',
  descriptors: [{
    id: '@deepseek-ai/dsh-refactory-console#refactoryConsole/runStream',
    service: 'refactoryConsole', namespace: 'refactoryConsole', method: 'runStream', mode: 'stream',
    invocation: { kind: 'direct' },
    parameters: [{ name: 'request', wire: 'request', source: 'json', codec: strictCodec('RefactorySingleRunRequest', request) }],
    cancellation: { parameter: 'signal' },
    result: strictCodec('RefactorySingleEvent', singleEvent),
  }],
}

/** Reconnectable, read-only observation of a durable run. */
export function watchFrame(value: unknown): RefactoryWatchFrame {
  const frame = record(value)
  const status = record(frame.status)
  const steering = record(frame.steering)
  if (!textFields(status, ['run_id', 'root', 'state']) || typeof status.cursor !== 'number'
    || !textFields(steering, ['run']) || !Array.isArray(steering.gates)) throw new Error('Invalid durable run frame')
  return frame as unknown as RefactoryWatchFrame
}
export const watchStreamContribution: TypertRemoteContribution = {
  package: '@deepseek-ai/dsh-refactory-console',
  descriptors: [{
    id: '@deepseek-ai/dsh-refactory-console#refactoryConsole/watchStream',
    service: 'refactoryConsole', namespace: 'refactoryConsole', method: 'watchStream', mode: 'stream',
    invocation: { kind: 'direct' },
    parameters: [
      { name: 'run', wire: 'run', source: 'json', codec: strictCodec('string', (value: unknown) => {
        if (typeof value !== 'string' || value.trim() === '') throw new Error('A run is required')
        return value
      }) },
      { name: 'after', wire: 'after', source: 'json', codec: strictCodec('number', (value: unknown) => {
        if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) throw new Error('Invalid cursor')
        return value
      }) },
    ],
    cancellation: { parameter: 'signal' },
    result: strictCodec('RefactoryWatchFrame', watchFrame),
  }],
}
