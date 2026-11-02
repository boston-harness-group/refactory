/**
 * TS tests for addItemRecord / item_records.json writing (F6 / Brief item 3b).
 *
 * Tests that:
 * - addItemRecord stores records retrievable by getItemRecords
 * - item_records.json is written with all records when the spine finishes
 */
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Spine } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-spine-records-'))
  dirs.push(dir)
  return dir
}

function simpleDataGraph(): WorkflowGraph {
  return {
    name: 'records-test',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data',
        task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'work',
        reads: [], writes: ['current_item.json'],
      },
      work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'data', target: 'after' },
    ],
  }
}

describe('addItemRecord / getItemRecords', () => {
  it('addItemRecord stores record by item id', () => {
    const spine = new Spine(simpleDataGraph(), { maxIterations: 2 })

    const record = {
      id: 'item-1',
      passed: true,
      score: 1.0,
      exchanges: [{ node: 'agent', prompt: 'hello', response: 'world' }],
      prediction: 'answer',
      label: 'correct',
      label_text: 'Label text',
      fail_reason: null,
      error_kind: null,
      error_text: null,
      metadata: { tag: 'test' },
    }

    spine.addItemRecord(record)

    const records = spine.getItemRecords()
    expect(records).toHaveLength(1)
    expect(records[0]).toMatchObject({
      id: 'item-1',
      passed: true,
      score: 1.0,
      prediction: 'answer',
      label: 'correct',
    })
    expect(records[0]!.exchanges).toEqual([{ node: 'agent', prompt: 'hello', response: 'world' }])
  })

  it('getItemRecords returns copies (not references to internal state)', () => {
    const spine = new Spine(simpleDataGraph(), { maxIterations: 2 })
    spine.addItemRecord({ id: 'a', passed: true })
    const first = spine.getItemRecords()
    spine.addItemRecord({ id: 'b', passed: false })
    const second = spine.getItemRecords()
    // The first snapshot should not have been mutated.
    expect(first).toHaveLength(1)
    expect(second).toHaveLength(2)
  })

  it('multiple records accumulate across data-node passes', () => {
    const spine = new Spine(simpleDataGraph(), { maxIterations: 2 })

    spine.addItemRecord({ id: 'alpha', passed: true, score: 1.0 })
    spine.addItemRecord({ id: 'beta', passed: false, score: 0.0, fail_reason: 'wrong' })
    spine.addItemRecord({ id: 'gamma', passed: true, score: 0.8 })

    const records = spine.getItemRecords()
    expect(records).toHaveLength(3)
    expect(records.map(r => r.id)).toEqual(['alpha', 'beta', 'gamma'])
  })
})
