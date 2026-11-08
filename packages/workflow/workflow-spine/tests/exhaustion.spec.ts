/**
 * A gate out of reloops does what its `on_exhausted` says (`Spine.verdict`): `continue`
 * takes its `proceed` edge, `fail_item` ends only the item it is in, and a gate without
 * the field halts the run as before. The walker does the same (`Executor._visit_gate`);
 * the Python suite runs one graph on both runtimes (`test_loop_exhaustion.py`).
 */
import { describe, expect, it } from 'vitest'
import { Spine } from '../src/spine.ts'
import type { WorkflowGraph, WorkflowNode } from '../src/types.ts'

/** data -> [solve -> check (reloop solve | proceed end)] -> after, the check capped at 1. */
function loopGraph(onExhausted?: WorkflowNode['on_exhausted']): WorkflowGraph {
  return {
    name: 'exhaustion',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data', task_ref: 'probe', subgraph_entry: 'solve',
        subgraph_exit: 'end', reads: [], writes: ['current_item.json'],
      },
      solve: { _type: 'FnNode', id: 'solve', command: 'true', reads: [], writes: [] },
      check: {
        _type: 'GateNode', id: 'check', evaluator_type: 'fn', reads: [], max_iterations: 1,
        ...(onExhausted === undefined ? {} : { on_exhausted: onExhausted }),
      },
      end: { _type: 'FnNode', id: 'end', command: 'true', reads: [], writes: [] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'data', target: 'after' },
      { source: 'solve', target: 'check' },
      { source: 'check', target: 'solve', condition: 'reloop' },
      { source: 'check', target: 'end', condition: 'proceed' },
    ],
  }
}

/** Open a two-item pass and spend the check's one reloop on the first item. */
function spentPass(spine: Spine): void {
  spine.startPass('data', 2)
  spine.complete('solve')
  spine.verdict('check', 'reloop', 'wrong once')
  spine.complete('solve')
}

describe('a gate out of reloops', () => {
  it('without on_exhausted halts the whole run, as before', () => {
    const spine = new Spine(loopGraph(), { maxIterations: 9 })
    spentPass(spine)
    const next = spine.verdict('check', 'reloop', 'wrong twice')
    expect(next).toMatchObject({ status: 'done', halted: { gate: 'check', reason: 'max_iterations' } })
    expect(spine.isHalted()).toBe(true)
    expect(spine.itemFailure()).toBeUndefined()
  })

  it('with continue takes its proceed edge and the item finishes with its last attempt', () => {
    const spine = new Spine(loopGraph('continue'), { maxIterations: 9 })
    spentPass(spine)
    const next = spine.verdict('check', 'reloop', 'wrong twice')
    expect(spine.isHalted()).toBe(false)
    expect(next.nodes?.map(node => node.id)).toEqual(['end'])
    expect(spine.state().gate_verdicts).toMatchObject({ check: 'proceed' })
    // Recorded as exhausted, unlike a proceed that passed.
    expect(spine.state().gate_exhausted).toEqual(['check'])
    // The refused reloop is not spent: the gate used its cap, as on the walker.
    expect(spine.state().iterations).toMatchObject({ check: 1 })
    spine.complete('end')
    // The second item starts with a fresh cap and the run reaches the data node's successor.
    expect(spine.endPass().nodes?.map(node => node.id)).toEqual(['solve'])
    spine.complete('solve')
    spine.verdict('check', 'reloop', 'wrong')
    spine.complete('solve')
    spine.verdict('check', 'proceed')
    // The second item's gate passed: nothing is recorded exhausted.
    expect(spine.state()).not.toHaveProperty('gate_exhausted')
    spine.complete('end')
    expect(spine.endPass().nodes?.map(node => node.id)).toEqual(['after'])
  })

  it('with fail_item in a sequential pass fails that item only, and the next item runs', () => {
    const spine = new Spine(loopGraph('fail_item'), { maxIterations: 9 })
    spentPass(spine)
    spine.verdict('check', 'reloop', 'still wrong')
    expect(spine.isHalted()).toBe(false)
    const failed = spine.itemFailure()
    expect(failed).toMatch(/gate check reached its max_iterations of 1 and its item fails/)
    expect(failed).toMatch(/still wrong/)
    // The pass is finished (its remaining node skipped), so the runner can judge it.
    expect(spine.passOf()).toMatchObject({ index: 0, finished: true, failed })
    expect(spine.state().skipped).toContain('end')
    // The reason was the first item's only.
    expect(spine.endPass().nodes?.map(node => node.id)).toEqual(['solve'])
    expect(spine.itemFailure()).toBeUndefined()
    expect(spine.passOf()).not.toHaveProperty('failed')
  })

  it('with fail_item on one item of a concurrent pass ends that item spine without a halt', () => {
    const spine = new Spine(loopGraph('fail_item'), { maxIterations: 9 })
    spine.startPass('data', 2)
    const item = spine.itemSpine()
    item.complete('solve')
    item.verdict('check', 'reloop', 'wrong')
    item.complete('solve')
    const next = item.verdict('check', 'reloop', 'wrong again')
    expect(next).toMatchObject({ status: 'done' })
    expect(next).not.toHaveProperty('halted')
    expect(item.haltOf()).toBeUndefined()
    expect(item.itemFailure()).toMatch(/item fails \(on_exhausted: fail_item\); last reason: wrong again/)
    // The pass's own spine is untouched: the other items go on.
    expect(spine.isHalted()).toBe(false)
    expect(spine.itemFailure()).toBeUndefined()
  })

  it('with continue on a concurrent item spine reaches the exit', () => {
    const spine = new Spine(loopGraph('continue'), { maxIterations: 9 })
    spine.startPass('data', 2)
    const item = spine.itemSpine()
    item.complete('solve')
    item.verdict('check', 'reloop', 'wrong')
    item.complete('solve')
    expect(item.verdict('check', 'reloop', 'wrong').nodes?.map(node => node.id)).toEqual(['end'])
    expect(item.complete('end')).toMatchObject({ status: 'done' })
    expect(item.haltOf()).toBeUndefined()
  })

  it('with continue whose proceed edge is its own retry halts rather than loop past the cap', () => {
    const graph = loopGraph('continue')
    const spine = new Spine({
      ...graph,
      edges: [
        ...graph.edges.filter(edge => edge.source !== 'check'),
        { source: 'check', target: 'solve', condition: 'proceed' },
        { source: 'check', target: 'end', condition: 'accept' },
      ],
    }, { maxIterations: 9 })
    spine.startPass('data', 1)
    spine.complete('solve')
    spine.verdict('check', 'proceed', 'again')
    spine.complete('solve')
    expect(spine.verdict('check', 'proceed', 'again')).toMatchObject({
      status: 'done', halted: { gate: 'check', reason: 'max_iterations' },
    })
  })

  it('with fail_item outside any pass halts the run (there is no item to fail)', () => {
    const graph = loopGraph('fail_item')
    const top: WorkflowGraph = {
      ...graph,
      start_node: 'solve',
      nodes: Object.fromEntries(Object.entries(graph.nodes).filter(([id]) => id !== 'data')),
      edges: [...graph.edges.filter(edge => edge.source !== 'data'), { source: 'end', target: 'after' }],
    }
    const spine = new Spine(top, { maxIterations: 9 })
    spine.complete('solve')
    spine.verdict('check', 'reloop', 'wrong')
    spine.complete('solve')
    expect(spine.verdict('check', 'reloop', 'wrong')).toMatchObject({
      status: 'done', halted: { gate: 'check', reason: 'max_iterations' },
    })
  })
})
