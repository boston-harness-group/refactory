/**
 * Graph shape invariant test suite.
 *
 * Drives named graph shapes through the spine in every completion order of the
 * nodes due together, checking five invariants on each run.
 */
import { describe, expect, it } from 'vitest'
import { shapes, crossContainerReloopGraph, designGraphRefineLightRelooped,
  designGraphDocumentReloop, designGraphApproachReloop } from './graph-shapes.ts'
import { allOrders, seededOrders, driveToEnd } from './shape-driver.ts'
import {
  checkCompletedOncePerVisit, checkJoinAfterBranches, checkNoDueDrop,
  checkDoneAndEnds, checkRedoInvariant,
} from './invariants.ts'
import { Spine, validateGraph } from '../src/spine.ts'

// ─── Structural validation ──────────────────────────────────────

describe('graph shapes validate', () => {
  for (const shape of shapes) {
    it(`${shape.name} passes validateGraph`, () => {
      expect(validateGraph(shape.graph)).toEqual([])
    })
  }
})

// ─── Invariants across all orderings ─────────────────────────────

describe('graph shape invariants', () => {
  for (const shape of shapes) {
    describe(shape.name, () => {
      const runs = shape.large ? seededOrders(shape, 42, 20) : allOrders(shape)

      runs.forEach((result, i) => {
        it(`ordering ${i}`, () => {
          checkCompletedOncePerVisit(result)
          checkJoinAfterBranches(result)
          checkNoDueDrop(result)
          checkDoneAndEnds(result, shape)
        })
      })

      // It replays the whole run once for each node the run completed, so its time grows
      // with the square of the run's length: design's run (its first self-check's fork
      // and gate among it) takes several seconds on a CI runner.
      it('redo invariant', () => {
        checkRedoInvariant(runs[0]!, shape)
      }, 60_000)
    })
  }
})

// ─── A light refine relooped at its approval gate ────────────────

describe('designGraphRefineLightRelooped', () => {
  const shape = designGraphRefineLightRelooped
  const research = ['research_director', 'record_findings', 'gate_research', 'classify_scale',
    'pitfalls_researcher', 'gate_pitfalls', 'route_strategy', 'strategy_director']

  seededOrders(shape, 42, 20).forEach((result, i) => {
    it(`ordering ${i}: the reloop replans and redesigns, and research never runs`, () => {
      checkCompletedOncePerVisit(result)
      checkJoinAfterBranches(result)
      checkNoDueDrop(result)
      checkDoneAndEnds(result, shape)
      const order = result.completedOrder
      const first = order.indexOf('gate_strategy')
      const second = order.indexOf('gate_strategy', first + 1)
      expect(order.slice(0, 3)).toEqual(['prepare', 'route_refine_design', 'design_doc'])
      expect(second).toBeGreaterThan(first)
      expect(order.slice(first + 1, second)).toEqual(
        ['classify_reloop', 'scope_reloop', 'synthesize_strategy', 'design_doc', 'check_design_claims', 'gate_design_review'])
      for (const skipped of research) expect(order).not.toContain(skipped)
    })
  })
})

// ─── Redo refusal for non-redoable types ─────────────────────────

describe('redo refusal for non-redoable types', () => {
  it('planRedo refuses GateNode with upstream guidance', () => {
    const shape = shapes.find(s => s.name === 'reloopBackEdge')!
    const spine = new Spine(shape.graph)
    spine.complete('emit')
    spine.complete('materialise')
    spine.verdict('check_gate', 'proceed', 'test')
    spine.complete('done')
    expect(() => spine.planRedo('check_gate')).toThrow(/gate/)
  })

  it('planRedo refuses DataNode', () => {
    const shape = shapes.find(s => s.name === 'dataTopLevel')!
    const spine = new Spine(shape.graph)
    spine.startPass('data', 2)
    spine.complete('work')
    spine.complete('d_exit')
    spine.endPass()
    spine.complete('work')
    spine.complete('d_exit')
    spine.endPass()
    spine.complete('after')
    expect(() => spine.planRedo('data')).toThrow(/iterates its items/)
  })
})

// ─── Invariant sanity checks ─────────────────────────────────────

describe('invariant sanity checks', () => {
  it('checkCompletedOncePerVisit rejects duplicates', () => {
    const shape = shapes.find(s => s.name === 'linear')!
    const result = driveToEnd(shape.graph, shape)
    // Create a fake spine state with a duplicate
    const fakeSpine = {
      state: () => ({ completed: ['a', 'b', 'a'], skipped: [] }),
      checkpoint: () => ({ graph: shape.graph }),
    } as unknown as Spine
    const fakeResult = { ...result, spine: fakeSpine }
    expect(() => checkCompletedOncePerVisit(fakeResult)).toThrow(/checkCompletedOncePerVisit/)
  })

  it('checkNoDueDrop rejects dropped node', () => {
    const shape = shapes.find(s => s.name === 'linear')!
    const result = driveToEnd(shape.graph, shape)
    // Inject a due node that was never completed
    const fakeEverDue = new Set([...result.everDue, 'phantom'])
    const fakeSpine = {
      state: () => ({ completed: ['a', 'b', 'c'], skipped: [] }),
      checkpoint: () => ({ graph: shape.graph }),
    } as unknown as Spine
    const fakeResult = { ...result, everDue: fakeEverDue, spine: fakeSpine }
    expect(() => checkNoDueDrop(fakeResult)).toThrow(/checkNoDueDrop/)
  })

  it('checkJoinAfterBranches rejects join before branch', () => {
    const shape = shapes.find(s => s.name === 'forkSingleBranch')!
    const result = driveToEnd(shape.graph, shape)
    // Swap join before its branch in completedOrder to trigger the invariant
    const idx = result.completedOrder.indexOf('join')
    const hcIdx = result.completedOrder.indexOf('hc')
    if (idx !== -1 && hcIdx !== -1) {
      const bad = [...result.completedOrder]
      bad[hcIdx] = 'join'
      bad[idx] = 'hc'
      const fakeResult = { ...result, completedOrder: bad }
      expect(() => checkJoinAfterBranches(fakeResult)).toThrow(/checkJoinAfterBranches/)
    }
  })

  it('checkDoneAndEnds rejects non-done status', () => {
    const shape = shapes.find(s => s.name === 'linear')!
    const result = driveToEnd(shape.graph, shape)
    const fakeResult = {
      ...result,
      final: { status: 'ready' as const, nodes: [] },
    }
    expect(() => checkDoneAndEnds(fakeResult, shape)).toThrow(/checkDoneAndEnds/)
  })
})

// ─── Cross-container reloop refusal ─────────────────────────────

describe('cross-container reloop refusal', () => {
  it('validateGraph reports cross-container reloop', () => {
    const issues = validateGraph(crossContainerReloopGraph)
    const msg = issues.map(i => i.message).join('; ')
    expect(msg).toMatch(/gate branch_gate is inside container fork.*reloop target start is outside/)
  })

  it('Spine constructor refuses cross-container reloop at load', () => {
    expect(() => new Spine(crossContainerReloopGraph)).toThrow(
      /gate branch_gate is inside container fork.*reloop target start is outside/,
    )
  })

  it('shipped design graph still loads', () => {
    const shape = shapes.find(s => s.name === 'designGraph')!
    expect(() => new Spine(shape.graph)).not.toThrow()
  })
})

// ─── Intra-branch reloop preserves sibling ──────────────────────

describe('intra-branch reloop preserves sibling', () => {
  it('reloop within a branch keeps sibling in cursor', () => {
    const shape = shapes.find(s => s.name === 'gateInsideBranch')!
    const spine = new Spine(shape.graph)
    spine.complete('start')
    spine.complete('fork')
    // cursor: [branch_work, other]
    spine.complete('branch_work')
    // cursor: [branch_gate, other]
    const state1 = spine.state() as { cursor: string[] }
    expect(state1.cursor).toContain('branch_gate')
    expect(state1.cursor).toContain('other')
    spine.verdict('branch_gate', 'reloop', 'fix it')
    // cursor should contain both branch_work (reloop target) and other (sibling)
    const state2 = spine.state() as { cursor: string[] }
    expect(state2.cursor).toContain('branch_work')
    expect(state2.cursor).toContain('other')
  })
})

// ─── Nested fork inner-join-before-outer-join ordering ──────────

describe('nested fork ordering', () => {
  it('inner join always completes before outer join in every completion order', () => {
    const shape = shapes.find(s => s.name === 'nestedFork')!
    const runs = allOrders(shape)
    for (const result of runs) {
      const innerIdx = result.completedOrder.indexOf('inner_join')
      const outerIdx = result.completedOrder.indexOf('outer_join')
      expect(innerIdx).toBeGreaterThan(-1)
      expect(outerIdx).toBeGreaterThan(-1)
      expect(innerIdx).toBeLessThan(outerIdx)
    }
  })
})

// ─── v1 checkpoint migration ────────────────────────────────────

describe('v1 checkpoint migration', () => {
  it('restores a v1 checkpoint with flat afters array and completes', () => {
    const shape = shapes.find(s => s.name === 'forkMultiBranch')!
    const spine = new Spine(shape.graph)
    spine.complete('start')
    spine.complete('fork')
    // Now cursor=[a1, b1], afters has {fork: 'join'}
    // Save as v2, then downgrade to v1 format to test migration
    const cp = JSON.parse(JSON.stringify(spine.checkpoint())) as Record<string, unknown>
    // Downgrade to v1 format
    const aftersMap = cp.afters_map as Record<string, string>
    cp.schema_version = 1
    cp.afters = Object.values(aftersMap)  // flat array: ['join']
    delete cp.afters_map
    // Restore from v1
    const spine2 = new Spine(shape.graph)
    spine2.restore(cp)
    // Drive to completion
    spine2.complete('a1')
    spine2.complete('b1')
    spine2.complete('a2')
    spine2.complete('b2')
    spine2.complete('join')
    spine2.complete('end')
    expect(spine2.next().status).toBe('done')
  })

  it('restores a v1 checkpoint with director successor in afters', () => {
    const shape = shapes.find(s => s.name === 'directorTopLevel')!
    const spine = new Spine(shape.graph)
    spine.expand('dir', shape.directors['dir']!)
    // Now cursor has expanded children, afters has {dir: 'after'}
    // Save as v2, then downgrade to v1 format
    const cp = JSON.parse(JSON.stringify(spine.checkpoint())) as Record<string, unknown>
    const aftersMap = cp.afters_map as Record<string, string>
    cp.schema_version = 1
    cp.afters = Object.values(aftersMap)  // flat array: ['after']
    delete cp.afters_map
    // Restore from v1
    const spine2 = new Spine(shape.graph)
    spine2.restore(cp)
    // Drive children to completion
    const state = spine2.state() as { cursor: string[] }
    for (const child of state.cursor) spine2.complete(child)
    // After all children, 'after' becomes due
    spine2.complete('after')
    expect(spine2.next().status).toBe('done')
  })
})

// ─── Data pass inside branch scoping ────────────────────────────

describe('data pass inside branch', () => {
  it('endPass scoping: pass finishes while sibling remains in cursor', () => {
    const shape = shapes.find(s => s.name === 'dataInsideBranch')!
    const spine = new Spine(shape.graph)
    spine.complete('start')
    spine.complete('fork')
    // cursor: [data_branch, other_branch]
    spine.startPass('data_branch', 1)
    // cursor should contain d_work and other_branch
    const s1 = spine.state() as { cursor: string[] }
    expect(s1.cursor).toContain('d_work')
    expect(s1.cursor).toContain('other_branch')
    spine.complete('d_work')
    spine.complete('d_exit')
    // pass.finished should be true, cursor still has other_branch
    const s2 = spine.state() as { cursor: string[] }
    expect(s2.cursor).toContain('other_branch')
    const next = spine.next()
    expect(next.status).toBe('ready')
    expect(next.pass?.finished).toBe(true)
    spine.endPass()
    // After endPass (last item with count=1), data_branch is completed
    // cursor should still have other_branch
    const s3 = spine.state() as { cursor: string[] }
    expect(s3.cursor).toContain('other_branch')
    spine.complete('other_branch')
    // join becomes due
    spine.complete('join')
    spine.complete('end')
    expect(spine.next().status).toBe('done')
  })
})

// ─── Ordering coverage ──────────────────────────────────────────

describe('ordering coverage', () => {
  it('forkSingleBranch has multiple orderings', () => {
    const shape = shapes.find(s => s.name === 'forkSingleBranch')!
    const runs = allOrders(shape)
    expect(runs.length).toBeGreaterThan(1)
  })

  it('linear has exactly 1 ordering', () => {
    const shape = shapes.find(s => s.name === 'linear')!
    const runs = allOrders(shape)
    expect(runs.length).toBe(1)
  })
})
