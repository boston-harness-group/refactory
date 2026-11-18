/**
 * The line a failed node's item error is named by (`nodeErrorLine`): a dispatched
 * worker's own line after its marker (`worker_node.main`), never a `worker:` line of a
 * message that quotes a model's reply.
 */
import { describe, expect, it } from 'vitest'
import { nodeErrorLine } from '../src/index.ts'

const MARKER = '--- refactory worker error ---'

describe('nodeErrorLine', () => {
  it('takes the worker line after the marker, not a quoted worker line above it', () => {
    const stderr = [
      'worker: parse failed: the reply was:',
      'worker: model session hit the harness backstop after 1s',
      MARKER,
      'ValueError: parse failed: the reply was:',
    ].join('\n')
    expect(nodeErrorLine(stderr)).toBe('ValueError: parse failed: the reply was:')
  })

  it('names a real backstop by its type', () => {
    const stderr = ['worker: model session hit the harness backstop after 9s', MARKER,
      'HarnessBackstop: model session hit the harness backstop after 9s'].join('\n')
    expect(nodeErrorLine(stderr)).toBe('HarnessBackstop: model session hit the harness backstop after 9s')
  })

  it('reads a marker quoted mid-stderr as no marker', () => {
    const stderr = ['worker: the reply said', MARKER, 'HarnessBackstop: fake', 'more log'].join('\n')
    expect(nodeErrorLine(stderr)).not.toBe('HarnessBackstop: fake')
  })
})
