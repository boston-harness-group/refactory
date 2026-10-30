/**
 * Reloop scope: after gate_strategy returns reloop, classify_reloop and
 * scope_reloop decide whether to re-derive the strategy or just revise
 * design.md.
 */
import { describe, expect, it } from 'vitest'
import {
  designGraphDocumentReloop,
  designGraphApproachReloop,
} from './graph-shapes.ts'
import { seededOrders, driveToEnd } from './shape-driver.ts'
import type { GatePolicy } from './shape-driver.ts'
import {
  checkCompletedOncePerVisit, checkJoinAfterBranches, checkNoDueDrop, checkDoneAndEnds,
} from './invariants.ts'

describe('reloop scope routing', () => {
  it('document-only reloop skips strategy synthesis', () => {
    const shape = designGraphDocumentReloop
    const runs = seededOrders(shape, 42, 10)
    for (const result of runs) {
      checkCompletedOncePerVisit(result)
      checkJoinAfterBranches(result)
      checkNoDueDrop(result)
      checkDoneAndEnds(result, shape)
      const order = result.completedOrder
      // synthesize_strategy never runs: the light refine skips it on the first pass,
      // and the document reloop goes straight to design_doc.
      const strategyCount = order.filter(n => n === 'synthesize_strategy').length
      expect(strategyCount).toBe(0)
      // After the first gate_strategy, the reloop goes through classify_reloop
      // and scope_reloop, then straight to design_doc (no synthesize_strategy)
      const first = order.indexOf('gate_strategy')
      const second = order.indexOf('gate_strategy', first + 1)
      expect(second).toBeGreaterThan(first)
      expect(order.slice(first + 1, second)).toEqual(
        ['classify_reloop', 'scope_reloop', 'design_doc', 'check_design_claims', 'gate_design_review'])
    }
  })

  it('approach reloop re-runs strategy synthesis', () => {
    const shape = designGraphApproachReloop
    const runs = seededOrders(shape, 42, 10)
    for (const result of runs) {
      checkCompletedOncePerVisit(result)
      checkJoinAfterBranches(result)
      checkNoDueDrop(result)
      checkDoneAndEnds(result, shape)
      const order = result.completedOrder
      // synthesize_strategy runs once: the light refine skips it on the first pass,
      // but the approach reloop runs it before design_doc.
      const strategyCount = order.filter(n => n === 'synthesize_strategy').length
      expect(strategyCount).toBe(1)
      // After the first gate_strategy, classify_reloop + scope_reloop + synthesize_strategy
      const first = order.indexOf('gate_strategy')
      const second = order.indexOf('gate_strategy', first + 1)
      expect(second).toBeGreaterThan(first)
      expect(order.slice(first + 1, second)).toEqual(
        ['classify_reloop', 'scope_reloop', 'synthesize_strategy', 'design_doc',
         'check_design_claims', 'gate_design_review'])
    }
  })

  it('reloop limit bounds the loop', () => {
    const shape = { ...designGraphApproachReloop, name: 'reloop-limit-test' }
    // A gate policy that always reloops gate_strategy
    const alwaysReloop: GatePolicy = (gateId, _verdicts, _encounter) => {
      if (gateId === 'gate_strategy') return 'reloop'
      if (shape.fixedVerdicts[gateId] !== undefined) return shape.fixedVerdicts[gateId]!
      return 'proceed'
    }
    const result = driveToEnd(shape.graph, shape, { gatePolicy: alwaysReloop })
    expect(result.final.status).toBe('done')
    // scope_reloop should have been visited up to the gate_strategy max_iterations (4)
    const scopeCount = result.completedOrder.filter(n => n === 'scope_reloop').length
    expect(scopeCount).toBe(4)
  })
})
