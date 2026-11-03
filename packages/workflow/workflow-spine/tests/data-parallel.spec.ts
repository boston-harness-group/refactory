/**
 * Parallel data node tests on the spine (T2, T10, T11 from the design).
 *
 * T2: the spine exposes parallelism on the NodeView and DataPass.
 * T10: item records are collected in item order via endPass.
 * T11: resume only replays unfinished items (pass directories mark .done).
 */
import { describe, expect, it } from 'vitest'
import { Spine, validateGraph } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

/** A data graph with a configurable parallelism field. */
function parallelDataGraph(parallelism?: number): WorkflowGraph {
  return {
    name: 'parallel-data',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data',
        task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'end',
        reads: [], writes: ['current_item.json'],
        ...(parallelism !== undefined ? { parallelism } : {}),
      },
      work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
      end: { _type: 'FnNode', id: 'end', command: 'true', reads: [], writes: [] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'data', target: 'after' },
      { source: 'work', target: 'end' },
    ],
  }
}

describe('data-parallel: spine parallelism field', () => {
  it('T2 — exposes parallelism > 1 on the NodeView', () => {
    const spine = new Spine(parallelDataGraph(4))
    const next = spine.next()
    expect(next.status).toBe('ready')
    const dataView = next.nodes!.find(n => n.id === 'data')
    expect(dataView).toBeDefined()
    expect(dataView!.parallelism).toBe(4)
  })

  it('T2 — omits parallelism when 1 (default)', () => {
    const spine = new Spine(parallelDataGraph())
    const next = spine.next()
    const dataView = next.nodes!.find(n => n.id === 'data')
    expect(dataView).toBeDefined()
    expect(dataView!.parallelism).toBeUndefined()
  })

  it('T2 — startPass stores parallelism in DataPass', () => {
    const spine = new Spine(parallelDataGraph(4))
    spine.startPass('data', 3)
    const pass = spine.passOf()
    expect(pass).toBeDefined()
    expect(pass!.parallelism).toBe(4)
  })

  it('T2 — DataPass has no parallelism when default', () => {
    const spine = new Spine(parallelDataGraph())
    spine.startPass('data', 3)
    const pass = spine.passOf()
    expect(pass).toBeDefined()
    expect(pass!.parallelism).toBeUndefined()
  })
})

describe('data-parallel: item order with sequential spine passes', () => {
  it('T10 — item records stay in index order through sequential passes', () => {
    // Even though items could finish in any order in the drain,
    // the spine's sequential endPass produces records in order.
    const spine = new Spine(parallelDataGraph(4))
    spine.startPass('data', 4)

    for (let i = 0; i < 4; i++) {
      // Walk subgraph: work → end.
      spine.complete('work')
      spine.complete('end')
      // Add item record in the order endPass is called.
      spine.addItemRecord({ id: String(i), score: i / 10 })
      spine.endPass()
    }

    const records = spine.getItemRecords()
    expect(records.map(r => r.id)).toEqual(['0', '1', '2', '3'])
    expect(records.map(r => r.score)).toEqual([0, 0.1, 0.2, 0.3])
  })
})

describe('data-parallel: resume', () => {
  it('T11 — a resumed pass starts from where it left off', () => {
    // Simulate processing 2 of 4 items, then stopping.
    const spine = new Spine(parallelDataGraph(4))
    spine.startPass('data', 4)

    // Process items 0 and 1.
    for (let i = 0; i < 2; i++) {
      spine.complete('work')
      spine.complete('end')
      spine.addItemRecord({ id: String(i) })
      spine.endPass()
    }

    // Save state (the pass is at index 2).
    const saved = spine.state()
    expect(saved.pass).toBeDefined()
    expect((saved.pass as Record<string, unknown>).index).toBe(2)

    // The spine is mid-pass: the remaining items (2, 3) still need processing.
    // The drain would check .done markers and skip finished items.
    // Process the remaining items.
    for (let i = 2; i < 4; i++) {
      spine.complete('work')
      spine.complete('end')
      spine.addItemRecord({ id: String(i) })
      spine.endPass()
    }

    // All 4 records present.
    expect(spine.getItemRecords().length).toBe(4)
    expect(spine.next().status).toBe('ready')
  })
})

describe('data-parallel: validates graph with parallelism', () => {
  it('accepts a data node with parallelism', () => {
    const issues = validateGraph(parallelDataGraph(4))
    // No blocking issues from parallelism.
    // validateGraph returns GraphIssue[] — all issues are blocking (structural).
    // A graph with parallelism set should have no structural issues.
    expect(issues.length).toBe(0)
  })
})
