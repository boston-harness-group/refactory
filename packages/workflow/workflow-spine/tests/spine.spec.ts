import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { Spine, validateGraph } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

function ids(spine: Spine): string[] {
  const next = spine.next()
  return next.nodes?.map(node => node.id) ?? []
}

describe('linear + gate', () => {
  const graph: WorkflowGraph = {
    name: 't',
    start_node: 'a',
    nodes: {
      a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
      gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', gate_prompt: 'ok?', reads: [] },
      b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
    },
    edges: [
      { source: 'a', target: 'gate' },
      { source: 'gate', target: 'b', condition: 'proceed' },
      { source: 'gate', target: 'a', condition: 'reloop' },
    ],
  }

  it('advances linearly through a gate', () => {
    const spine = new Spine(graph)
    expect(ids(spine)).toEqual(['a'])
    spine.complete('a')
    expect(ids(spine)).toEqual(['gate'])
    spine.verdict('gate', 'proceed')
    expect(ids(spine)).toEqual(['b'])
    spine.complete('b')
    expect(spine.next().status).toBe('done')
  })

  it('reloops with feedback and enforces the iteration cap', () => {
    const spine = new Spine(graph)
    spine.complete('a')
    spine.verdict('gate', 'reloop', 'try harder')
    expect(ids(spine)).toEqual(['a'])
    expect(spine.state().feedback).toEqual({ a: 'try harder' })
    // The cap is three reloops: the first verdict above is one, these three take
    // it past the cap, and the run halts rather than looping forever.
    for (let index = 0; index < 3; index += 1) {
      spine.complete('a')
      spine.verdict('gate', 'reloop', 'nope')
    }
    expect(spine.next().status).toBe('done')
    expect(spine.state().gate_verdicts).toEqual({ gate: 'halt' })
  })

  it('drops a reloop target\'s feedback once the target has run again', () => {
    const spine = new Spine(graph)
    spine.complete('a')
    spine.verdict('gate', 'reloop', 'try harder')
    expect(spine.next().feedback).toEqual({ a: 'try harder' })
    // The feedback was for the rerun. Once that has happened, later views stop
    // carrying it, however long it was.
    spine.complete('a')
    expect(spine.next().feedback).toEqual({})
  })

  it('keeps feedback for a command target, which never reads it, for the driver to see', () => {
    const spine = new Spine({
      ...graph,
      nodes: { ...graph.nodes, a: { _type: 'FnNode', id: 'a', command: 'run-a', reads: [], writes: [] } },
    })
    spine.complete('a')
    spine.verdict('gate', 'reloop', 'try harder')
    spine.complete('a')
    expect(spine.next().feedback).toEqual({ a: 'try harder' })
  })
})

describe('director expansion', () => {
  const graph: WorkflowGraph = {
    name: 't',
    start_node: 'dir',
    nodes: {
      dir: {
        _type: 'AgentNode', id: 'dir', role: 'ceo', prompt_template: 'plan',
        expand: { plan: 'p.json', role: 'researcher', slug_field: 'slug', prompt_field: 'prompt', writes_template: 'research-{slug}.md' },
      },
      after: { _type: 'FnNode', id: 'after', writes: [], command: '' },
    },
    edges: [{ source: 'dir', target: 'after' }],
  }

  it('expands children then advances to the successor', () => {
    const spine = new Spine(graph)
    spine.expand('dir', [{ slug: 'x', prompt: 'p1' }, { slug: 'y', prompt: 'p2' }])
    expect(ids(spine)).toEqual(['dir__x', 'dir__y'])
    spine.complete('dir__x')
    spine.complete('dir__y')
    expect(ids(spine)).toEqual(['after'])
  })

  it('re-expanding reverts the previous children', () => {
    const spine = new Spine(graph)
    spine.expand('dir', [{ slug: 'x', prompt: 'p1' }])
    spine.complete('dir__x')
    spine.complete('after')
    spine.expand('dir', [{ slug: 'z', prompt: 'p2' }])
    expect(ids(spine)).toEqual(['dir__z'])
    expect(spine.state()).not.toHaveProperty('dir__x')
  })
})

describe('fork / join', () => {
  const graph: WorkflowGraph = {
    name: 't',
    start_node: 'fork',
    nodes: {
      fork: { _type: 'ForkNode', id: 'fork', targets: ['c1', 'c2'] },
      c1: { _type: 'AgentNode', id: 'c1', role: 'reviewer', prompt_template: 'r1', writes: [], reads: [] },
      c2: { _type: 'AgentNode', id: 'c2', role: 'reviewer', prompt_template: 'r2', writes: [], reads: [] },
      join: { _type: 'JoinNode', id: 'join', sources: ['c1', 'c2'] },
      end: { _type: 'FnNode', id: 'end', writes: [], command: '' },
    },
    edges: [
      { source: 'fork', target: 'join' },
      { source: 'join', target: 'end' },
    ],
  }

  it('fans out in parallel then barriers at the join', () => {
    const spine = new Spine(graph)
    expect(ids(spine)).toEqual(['fork'])
    spine.complete('fork')
    expect(ids(spine)).toEqual(['c1', 'c2'])
    spine.complete('c1')
    expect(ids(spine)).toEqual(['c2'])
    spine.complete('c2')
    expect(ids(spine)).toEqual(['join'])
    spine.complete('join')
    expect(ids(spine)).toEqual(['end'])
  })
})

describe('four-branch fork with optional verifier', () => {
  it('waits for all four branches including optional verifier before join', () => {
    const graph: WorkflowGraph = {
      name: 'test_four_branch', start_node: 'start',
      nodes: {
        start:           { _type: 'FnNode', id: 'start' },
        fork_qa:         { _type: 'ForkNode', id: 'fork_qa', targets: ['hc', 'cr', 'qd', 'verifier'], optional_targets: ['verifier'] },
        hc:              { _type: 'AgentNode', id: 'hc', role: 'ceo', prompt_template: '' },
        cr:              { _type: 'AgentNode', id: 'cr', role: 'ceo', prompt_template: '' },
        qd:              { _type: 'AgentNode', id: 'qd', role: 'ceo', prompt_template: '' },
        verifier:        { _type: 'AgentNode', id: 'verifier', role: 'code_reviewer', prompt_template: '' },
        join_qa:         { _type: 'JoinNode', id: 'join_qa', sources: ['hc', 'cr', 'qd'] },
        gate_qa:         { _type: 'GateNode', id: 'gate_qa', evaluator_type: 'fn', max_iterations: 5 },
      },
      edges: [
        { source: 'start', target: 'fork_qa' },
        { source: 'fork_qa', target: 'join_qa' },
        { source: 'verifier', target: 'join_qa' },
        { source: 'join_qa', target: 'gate_qa' },
      ],
    }
    const s = new Spine(graph)
    expect(ids(s)).toEqual(['start'])
    s.complete('start')
    expect(ids(s)).toEqual(['fork_qa'])
    s.complete('fork_qa')
    // All four branches appear
    expect(ids(s).sort()).toEqual(['cr', 'hc', 'qd', 'verifier'])
    // Complete three reviewers — verifier still pending, join not reached
    s.complete('hc')
    s.complete('cr')
    s.complete('qd')
    expect(ids(s)).toContain('verifier')
    expect(ids(s)).not.toContain('join_qa')
    // Complete verifier (simulating "missing" status at the app level)
    s.complete('verifier')
    expect(ids(s)).toEqual(['join_qa'])
    s.complete('join_qa')
    expect(ids(s)).toEqual(['gate_qa'])
    // Check optional_targets is visible
    const forkView = s.view('fork_qa')
    expect(forkView.optional_targets).toEqual(['verifier'])
  })
})

describe('design-graph fixture fork_qa carries optional_targets', () => {
  const graph = JSON.parse(
    readFileSync(join(__dirname, 'fixtures', 'design-graph.json'), 'utf8'),
  ) as WorkflowGraph

  it('fork_qa in the regenerated fixture has optional_targets: ["verifier"]', () => {
    const s = new Spine(graph)
    const forkView = s.view('fork_qa')
    expect(forkView.optional_targets).toEqual(['verifier'])
    expect(forkView.targets).toEqual(
      expect.arrayContaining(['health_checker', 'code_reviewer', 'qa_director', 'verifier']),
    )
  })

  /** A node's out edges in the fixture, as [target, condition] pairs, sorted. */
  function spineEdges(g: WorkflowGraph, source: string): Array<[string, string]> {
    return g.edges.filter(e => e.source === source)
      .map(e => [e.target, e.condition ?? ''] as [string, string]).sort()
  }

  /** Walk to `until`, answering the scale gates with `scale` and every other gate proceed.
   *  route_refine_design always gets 'proceed' because these tests test the full design path. */
  function advance(s: Spine, until: string, scale: 'proceed' | 'scoped'): string[] {
    const visited: string[] = []
    while (!ids(s).includes(until)) {
      for (const n of ids(s)) {
        visited.push(n)
        const view = s.view(n)
        if (view._type === 'GateNode') {
          if (n === 'route_refine_design') s.verdict(n, 'proceed')
          else s.verdict(n, n.startsWith('route_') ? scale : 'proceed')
        }
        else s.complete(n)
      }
    }
    return visited
  }

  it('a scoped change skips the perspectives step, and no reviewer runs at either scale', () => {
    const scoped = advance(new Spine(graph), 'emit', 'scoped')
    expect(scoped).toContain('synthesize_strategy')
    expect(scoped).toContain('gate_design_review')
    expect(scoped).not.toContain('strategy_director')
    const full = advance(new Spine(graph), 'emit', 'proceed')
    expect(full).toContain('strategy_director')
    for (const visited of [scoped, full]) {
      const design = visited.indexOf('design_doc')
      expect(visited.slice(design, visited.indexOf('gate_strategy'))).toEqual(
        ['design_doc', 'check_design_claims', 'gate_design_review'])
    }
  })

  it('the check runs beside the reviews: five branches drain before join_qa, then gate_check routes', () => {
    const s = new Spine(graph)
    // Advance through the graph until fork_qa; prepare_check settles the commit first.
    const before = advance(s, 'fork_qa', 'proceed')
    expect(before).toContain('prepare_check')
    expect(before).not.toContain('check')
    s.complete('fork_qa')
    expect(ids(s)).toEqual(expect.arrayContaining(
      ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']))
    // The check records its decision and completes into the join; it may not leave the fork.
    s.complete('check')
    s.complete('health_checker')
    s.complete('code_reviewer')
    s.complete('qa_director')
    expect(ids(s)).toContain('verifier')
    expect(ids(s)).not.toContain('join_qa')
    // Complete verifier (simulating "missing" status at the app level)
    s.complete('verifier')
    expect(ids(s)).toContain('join_qa')
    s.complete('join_qa')
    expect(ids(s)).toEqual(['gate_check'])
    s.verdict('gate_check', 'proceed')
    expect(ids(s)).toContain('synthesize_qa')
    s.complete('synthesize_qa')
    // The QA judge reads the merged reviews and writes the verdict gate_qa routes on.
    expect(ids(s)).toEqual(['qa_judge'])
    s.complete('qa_judge')
    expect(ids(s)).toContain('gate_qa')
  })

  it('the judges decide: the correction audit runs between gate_qa and the overwatch, and gate_overwatch has no audit round', () => {
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    s.verdict('gate_check', 'proceed')
    s.complete('synthesize_qa')
    s.complete('qa_judge')
    s.verdict('gate_qa', 'proceed')
    expect(ids(s)).toEqual(['audit_corrections'])
    s.complete('audit_corrections')
    expect(ids(s)).toEqual(['overwatch'])
    s.complete('overwatch')
    expect(ids(s)).toEqual(['gate_overwatch'])
    expect(spineEdges(graph, 'gate_overwatch')).toEqual(
      [['overwatch', 'reread'], ['pick_corrections', 'reloop'], ['report', 'proceed']])
    s.verdict('gate_overwatch', 'reloop', 'the overwatch sent it back')
    toTheBuilder(s)
  })

  it('a judge whose verdict could not be read is asked again, and only the judge runs', () => {
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    s.verdict('gate_check', 'proceed')
    s.complete('synthesize_qa')
    s.complete('qa_judge')
    s.verdict('gate_qa', 'reread', 'qa-judge.json holds no JSON object')
    expect(ids(s)).toEqual(['qa_judge'])
    s.complete('qa_judge')
    s.verdict('gate_qa', 'proceed')
    s.complete('audit_corrections')
    s.complete('overwatch')
    s.verdict('gate_overwatch', 'reread', 'overwatch-verdict.json holds no JSON object')
    expect(ids(s)).toEqual(['overwatch'])
    s.complete('overwatch')
    s.verdict('gate_overwatch', 'proceed')
    expect(ids(s)).toEqual(['report'])
  })

  it('a QA judge redo sends the build back to emit', () => {
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    s.verdict('gate_check', 'proceed')
    s.complete('synthesize_qa')
    s.complete('qa_judge')
    s.verdict('gate_qa', 'reloop', 'the QA judge sent it back')
    toTheBuilder(s)
  })

  it('a failed check, routed after the join, sends the build back to emit', () => {
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    s.verdict('gate_check', 'reloop', 'the check failed')
    toTheBuilder(s)
  })

  it('a check still failing at its limit goes from gate_check straight to the report, as a draft', () => {
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    expect(spineEdges(graph, 'gate_check')).toEqual(
      [['pick_corrections', 'reloop'], ['report', 'draft'], ['synthesize_qa', 'proceed']])
    s.verdict('gate_check', 'draft', 'check attempts exhausted after 3 check failure(s)')
    expect(ids(s)).toEqual(['report'])
  })

  it('a limit stop before the review fork goes from prepare_check straight to the report, as a draft', () => {
    const s = new Spine(graph)
    advance(s, 'prepare_check', 'proceed')
    expect(spineEdges(graph, 'prepare_check')).toEqual(
      [['fork_qa', 'proceed'], ['fork_self_check', 'first'], ['pick_corrections', 'reloop'], ['report', 'draft']])
    s.verdict('prepare_check', 'draft', 'emissions refused 3 consecutive times')
    expect(ids(s)).toEqual(['report'])
  })

  /** A send-back reaches the builder through pick_corrections (dm-226): with no correction
   *  given since the approval it skips straight to emit. */
  function toTheBuilder(s: Spine): void {
    expect(ids(s)).toEqual(['pick_corrections'])
    // The send-back reopened the build: emit's rerun is a completion of its own.
    expect(s.state().completed).not.toContain('emit')
    expect(s.state().completed).not.toContain('apply_corrections')
    s.verdict('pick_corrections', 'skip')
    expect(ids(s)).toEqual(['emit'])
  }

  it('every send-back to the builder goes through pick_corrections, which applies a correction given since', () => {
    for (const gate of ['prepare_check', 'gate_self_check', 'gate_check', 'gate_qa', 'gate_overwatch', 'delivery_gate']) {
      expect(spineEdges(graph, gate)).toContainEqual(['pick_corrections', 'reloop'])
    }
    expect(graph.edges.filter(e => e.target === 'emit').map(e => [e.source, e.condition ?? '']).sort())
      .toEqual([['apply_corrections', ''], ['pick_corrections', 'skip']])
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    s.verdict('gate_check', 'proceed')
    s.complete('synthesize_qa')
    s.complete('qa_judge')
    s.verdict('gate_qa', 'proceed')
    s.complete('audit_corrections')
    s.complete('overwatch')
    s.verdict('gate_overwatch', 'reloop', 'a correction given after the approval is not applied yet')
    expect(ids(s)).toEqual(['pick_corrections'])
    s.verdict('pick_corrections', 'proceed')
    expect(ids(s)).toEqual(['apply_corrections'])
    s.complete('apply_corrections')
    expect(ids(s)).toEqual(['emit'])
  })

  it('a gate_check send-back reopens the whole round, and a delivery_gate one the report and delivery', () => {
    const s = new Spine(graph)
    advance(s, 'fork_qa', 'proceed')
    s.complete('fork_qa')
    for (const n of ['check', 'health_checker', 'code_reviewer', 'qa_director', 'verifier']) s.complete(n)
    s.complete('join_qa')
    s.verdict('gate_check', 'reloop', 'the check failed')
    const reopened = s.state().completed as string[]
    for (const n of ['fork_qa', 'code_reviewer', 'check', 'join_qa', 'emit', 'materialise']) {
      expect(reopened).not.toContain(n)
    }
    expect(reopened).toContain('gate_strategy')
    // The send-back's reason is pick_corrections' visit feedback until it decides.
    expect(s.state().feedback).toEqual(expect.objectContaining({ pick_corrections: 'the check failed' }))
    s.verdict('pick_corrections', 'skip')
    expect(s.state().feedback).not.toHaveProperty('pick_corrections')

    const d = new Spine(graph)
    advance(d, 'delivery_gate', 'proceed')
    expect(d.state().completed).toEqual(expect.arrayContaining(['report', 'deliver']))
    d.verdict('delivery_gate', 'reloop', 'conflicts with main')
    expect(d.state().completed).not.toContain('report')
    expect(d.state().completed).not.toContain('deliver')
    expect(d.state().completed).toContain('gate_strategy')
  })

  it('a send-back onto a gate reopens its cycle through nested gates, but not what its own retry reaches', () => {
    const nested = {
      name: 'nested', start_node: 'S',
      nodes: Object.fromEntries([
        ...['S', 'A', 'B', 'E'].map(id => [id, { _type: 'FnNode', id, command: 'true' }]),
        ...['H', 'H2', 'X'].map(id => [id, { _type: 'GateNode', id, evaluator_type: 'fn', evaluator_command: 'true', max_iterations: 3 }]),
      ]),
      edges: [['S', 'H'], ['H', 'H2', 'proceed'], ['H', 'B', 'skip'], ['H', 'S', 'reloop'], ['H2', 'A', 'a'],
        ['H2', 'B', 'b'], ['A', 'B'], ['B', 'X'], ['X', 'H', 'reloop'], ['X', 'E', 'proceed']]
        .map(([source, target, condition]) => ({ source, target, ...(condition ? { condition } : {}) })),
    } as unknown as WorkflowGraph
    const s = new Spine(nested)
    s.complete('S')
    s.verdict('H', 'proceed')
    s.verdict('H2', 'a')
    s.complete('A')
    s.complete('B')
    // E, reachable from H but outside the cycle (it cannot reach X), completed before.
    const saved = JSON.parse(JSON.stringify(s.checkpoint()))
    saved.state.completed = [...saved.state.completed, 'E']
    const t = new Spine(nested)
    t.restore(saved)
    t.verdict('X', 'reloop', 'again')
    const completed = t.state().completed as string[]
    for (const n of ['H', 'H2', 'A', 'B']) expect(completed).not.toContain(n)
    expect(completed).toEqual(expect.arrayContaining(['S', 'E']))
    expect(ids(t)).toEqual(['H'])
  })

  /** One build round from emit to the gate after the first self-check's fork (b48a). */
  function firstSelfCheck(s: Spine): void {
    s.complete('emit')
    s.complete('materialise')
    s.verdict('prepare_check', 'first')
    expect(ids(s)).toEqual(['fork_self_check'])
    s.complete('fork_self_check')
    // The whole check and the early asks judge run at once.
    expect(ids(s).sort()).toEqual(['early_asks_check', 'first_self_check'])
    expect(s.view('early_asks_check')._type).toBe('AgentNode')
    s.complete('first_self_check')
    expect(ids(s)).toEqual(['early_asks_check'])
    s.complete('early_asks_check')
    s.complete('join_self_check')
    expect(ids(s)).toEqual(['gate_self_check'])
  }

  it('an attempt\'s first self-check runs beside the early asks judge, and a fix-up\'s runs alone', () => {
    const s = new Spine(graph)
    advance(s, 'emit', 'proceed')
    expect(spineEdges(graph, 'gate_self_check')).toEqual(
      [['fork_qa', 'proceed'], ['fork_self_check', 'reread'], ['pick_corrections', 'reloop'], ['report', 'draft']])
    firstSelfCheck(s)
    s.verdict('gate_self_check', 'reloop', 'self-check fix-up 1 of 2: the early asks judge found 1 ask(s) or correction(s) not built')
    toTheBuilder(s)
    // The fix-up's self-check runs in prepare_check: no fork, no judge.
    s.complete('emit')
    s.complete('materialise')
    s.verdict('prepare_check', 'proceed')
    expect(ids(s)).toEqual(['fork_qa'])
  })

  it('a judge asked again reruns only the fork, and the gate never ends the run however often its pair is used', () => {
    const s = new Spine(graph)
    advance(s, 'emit', 'proceed')
    let judged = 0
    for (let attempt = 0; attempt < 8; attempt += 1) {
      firstSelfCheck(s)
      judged += 1
      // The judge's verdict could not be read: the fork runs again, the gate is due again.
      s.verdict('gate_self_check', 'reread', 'the early asks judge\'s verdict could not be read')
      expect(ids(s)).toEqual(['fork_self_check'])
      s.complete('fork_self_check')
      s.complete('first_self_check')
      s.complete('early_asks_check')
      s.complete('join_self_check')
      judged += 1
      s.verdict('gate_self_check', 'reloop', 'self-check fix-up 1 of 2')
      expect(s.isHalted()).toBe(false)
      toTheBuilder(s)
      // The fix-up's self-check spends the attempt (the early stop), back to emit.
      s.complete('emit')
      s.complete('materialise')
      s.verdict('prepare_check', 'reloop', 'check attempts: the early stop')
      expect(s.isHalted()).toBe(false)
      toTheBuilder(s)
    }
    expect(judged).toBe(16)
    expect(s.reloops('gate_self_check')).toEqual({ used: 16, cap: 91, target: 'pick_corrections' })
    firstSelfCheck(s)
    s.verdict('gate_self_check', 'proceed')
    expect(ids(s)).toEqual(['fork_qa'])
  })
})

describe('skip mutation', () => {
  const graph: WorkflowGraph = {
    name: 't',
    start_node: 'fork',
    nodes: {
      fork: { _type: 'ForkNode', id: 'fork', targets: ['c1'] },
      c1: { _type: 'AgentNode', id: 'c1', role: 'reviewer', prompt_template: 'r', writes: [], reads: [] },
      join: { _type: 'JoinNode', id: 'join', sources: ['c1'] },
      end: { _type: 'FnNode', id: 'end', writes: [], command: '' },
    },
    edges: [
      { source: 'fork', target: 'join' },
      { source: 'join', target: 'end' },
    ],
  }

  it('skipping a fork hot-removes its subtree and jumps to the join', () => {
    const spine = new Spine(graph)
    spine.skip('fork')
    expect(ids(spine)).toEqual(['join'])
    // Only the explicitly skipped node is recorded; its subtree is bypassed
    // rather than completed, so it is no longer due.
    expect(spine.state().skipped).toEqual(['fork'])
    expect(() => spine.complete('c1')).toThrow(/node c1 is not due/)
    expect(spine.state().completed).not.toContain('c1')
  })

  it('skipping the last parallel entry releases the barrier behind it', () => {
    const spine = new Spine(graph)
    spine.complete('fork')
    spine.skip('c1')
    // The join was waiting behind the barrier; emptying the cursor releases it.
    expect(ids(spine)).toEqual(['join'])
    expect(spine.state().skipped).toContain('c1')
  })

  it('unskips a node, and is inert for one that was never skipped', () => {
    const spine = new Spine(graph)
    spine.skip('c1')
    expect(spine.state().skipped).toContain('c1')
    spine.unskip('c1')
    expect(spine.state().skipped).not.toContain('c1')
    expect(() => spine.unskip('c1')).not.toThrow()
  })

  it('refuses a node that is not in the graph', () => {
    const spine = new Spine(graph)
    expect(() => spine.skip('absent')).toThrow(/unknown node absent/)
  })
})

describe('routing a gate', () => {
  const graph: WorkflowGraph = {
    name: 't',
    start_node: 'gate',
    nodes: {
      gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', gate_prompt: 'ok?', reads: [] },
      back: { _type: 'FnNode', id: 'back', writes: [], command: '' },
      on: { _type: 'FnNode', id: 'on', writes: [], command: '' },
    },
    edges: [
      { source: 'gate', target: 'back', condition: 'reloop' },
      { source: 'gate', target: 'on' },
    ],
  }

  it('falls through the unconditional edge on proceed when the gate declares no proceed label', () => {
    const spine = new Spine(graph)
    spine.verdict('gate', 'proceed')
    expect(ids(spine)).toEqual(['on'])
  })

  it('names the labels it does accept for an outcome it does not have', () => {
    const spine = new Spine(graph)
    expect(() => spine.verdict('gate', 'maybe')).toThrow(/has no 'maybe' edge \(accepts reloop\)/)
  })

  it('halts to the halt edge, and ends the run when the gate declares none', () => {
    const withHalt = new Spine({
      name: 't',
      start_node: 'gate',
      nodes: {
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] },
        stop: { _type: 'FnNode', id: 'stop', writes: [], command: '' },
      },
      edges: [{ source: 'gate', target: 'stop', condition: 'halt' }],
    })
    withHalt.verdict('gate', 'halt')
    expect(ids(withHalt)).toEqual(['stop'])

    const withoutHalt = new Spine({
      name: 't',
      start_node: 'gate',
      nodes: {
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] },
        on: { _type: 'FnNode', id: 'on', writes: [], command: '' },
      },
      edges: [{ source: 'gate', target: 'on', condition: 'proceed' }],
    })
    withoutHalt.verdict('gate', 'halt')
    expect(withoutHalt.next().status).toBe('done')
  })
})

describe('a selection node is described, not executed', () => {
  it('reports its strategy and the files it reads and writes', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'pick',
      nodes: {
        pick: { _type: 'SelectionNode', id: 'pick', strategy: 'best_score', reads: ['in.json'], writes: ['out.json'] },
      },
      edges: [],
    })
    const view = spine.view('pick')
    expect(view.strategy).toBe('best_score')
    expect(view.reads).toEqual(['in.json'])
    expect(view.writes).toEqual(['out.json'])
  })
})

describe('director expansion edges', () => {
  it('refuses to expand a director that declares no expansion', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'a',
      nodes: { a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] } },
      edges: [],
    })
    expect(() => spine.expand('a', [])).toThrow(/has no expand annotation/)
  })

  it('re-points a join that named the director when the director has no successor', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'dir',
      nodes: {
        dir: {
          _type: 'AgentNode', id: 'dir', role: 'ceo', prompt_template: 'plan',
          expand: { plan: 'p.json', role: 'researcher', slug_field: 'slug', prompt_field: 'prompt', writes_template: '' },
        },
        join: { _type: 'JoinNode', id: 'join', sources: ['dir'] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: '' },
      },
      edges: [{ source: 'join', target: 'end' }],
    }
    const spine = new Spine(graph)
    spine.expand('dir', [{ slug: 'x', prompt: 'p1' }])
    // The view is what the CEO reads, and the join now waits on the children.
    expect(spine.view('join').sources).toEqual(['dir__x'])

    // Re-expanding reverts the join's reference to the previous children first.
    spine.expand('dir', [{ slug: 'y', prompt: 'p2' }])
    expect(spine.view('join').sources).toEqual(['dir__y'])
  })

  it('leaves a join that declares no sources alone, and keeps a director source when a successor exists', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'dir',
      nodes: {
        dir: {
          _type: 'AgentNode', id: 'dir', role: 'ceo', prompt_template: 'plan',
          expand: { plan: 'p.json', role: 'researcher', slug_field: 'slug', prompt_field: 'prompt', writes_template: '' },
        },
        bare: { _type: 'JoinNode', id: 'bare' },
        waiting: { _type: 'JoinNode', id: 'waiting', sources: ['dir'] },
      },
      edges: [{ source: 'dir', target: 'after' }],
    })
    spine.expand('dir', [{ slug: 'x', prompt: 'p1' }])
    expect(spine.view('bare').sources).toEqual([])
    // The successor is reached through the barrier, so the director keeps the join.
    expect(spine.view('waiting').sources).toEqual(['dir'])
  })

  it('names a child with no slug or prompt rather than dropping it', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'dir',
      nodes: {
        dir: {
          _type: 'AgentNode', id: 'dir', role: 'ceo', prompt_template: 'plan',
          expand: { plan: 'p.json', role: 'researcher', slug_field: 'slug', prompt_field: 'prompt', writes_template: '' },
        },
      },
      edges: [],
    })
    const result = spine.expand('dir', [{}, { slug: 'a b/c' }])
    expect(result.children).toEqual(['dir__child0', 'dir__a-b-c'])
    expect(spine.view('dir__child0').prompt_template).toBe('')
  })
})

describe('describing the remaining node types', () => {
  it('reports an LLMNode as the one-shot call it is', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'llm',
      nodes: {
        llm: {
          _type: 'LLMNode', id: 'llm', system_prompt: 'summarise', model: 'flash',
          temperature: 0.5, reads: ['in'], writes: ['out'],
        },
      },
      edges: [],
    })
    const view = spine.view('llm')
    expect(view.prompt_template).toBe('summarise')
    expect(view.model).toBe('flash')
    expect(view.temperature).toBe(0.5)
    expect(view.writes).toEqual(['out'])
  })

  it('reports a fork\'s targets and a join\'s sources', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['a', 'b'] },
        a: { _type: 'FnNode', id: 'a', writes: [], command: '' },
        b: { _type: 'FnNode', id: 'b', writes: [], command: '' },
        join: { _type: 'JoinNode', id: 'join', sources: ['a', 'b'] },
      },
      edges: [{ source: 'fork', target: 'join' }],
    })
    expect(spine.view('fork').targets).toEqual(['a', 'b'])
    expect(spine.view('join').sources).toEqual(['a', 'b'])
  })
})

describe('validation', () => {
  it('rejects a node keyed under a name it does not declare', () => {
    const issues = validateGraph({
      name: 't',
      start_node: 'a',
      nodes: { a: { _type: 'FnNode', id: 'b', writes: [], command: 'true' } },
      edges: [],
    })
    expect(issues.map(issue => issue.message)).toContain("node declares id 'b' but is keyed as 'a'")
  })

  it('rejects a gate with two unconditional edges', () => {
    const issues = validateGraph({
      name: 't',
      start_node: 'gate',
      nodes: {
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        a: { _type: 'FnNode', id: 'a', writes: [], command: '' },
        b: { _type: 'FnNode', id: 'b', writes: [], command: '' },
      },
      edges: [{ source: 'gate', target: 'a' }, { source: 'gate', target: 'b' }],
    })
    expect(issues.map(issue => issue.message)).toContain('gate has more than one unconditional edge')
  })
})

describe('rewinding across a fork', () => {
  it('un-completes the fork\'s branches when a gate reloops past it', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', writes: [], command: '' },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['c1', 'c2'] },
        c1: { _type: 'AgentNode', id: 'c1', role: 'reviewer', prompt_template: 'r1', writes: [], reads: [] },
        c2: { _type: 'AgentNode', id: 'c2', role: 'reviewer', prompt_template: 'r2', writes: [], reads: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['c1', 'c2'] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: '' },
      },
      edges: [
        { source: 'a', target: 'fork' },
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'end', condition: 'proceed' },
      ],
    })
    spine.complete('a')
    spine.complete('fork')
    spine.complete('c1')
    spine.complete('c2')
    spine.complete('join')
    expect(ids(spine)).toEqual(['gate'])
    spine.verdict('gate', 'reloop', 'again')
    // The rewind un-completed the whole branch, not only the node it targeted.
    expect(spine.state().completed).not.toContain('c1')
    expect(spine.state().completed).not.toContain('c2')
    expect(ids(spine)).toEqual(['a'])
  })

  it('rewinds through a fork that declares no targets, and one whose branch is the gate', () => {
    const bareFork: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', writes: [], command: 'true' },
        fork: { _type: 'ForkNode', id: 'fork' },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: 'true' },
      },
      edges: [
        { source: 'a', target: 'fork' },
        { source: 'fork', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'end', condition: 'proceed' },
      ],
    }
    const bare = new Spine(bareFork)
    bare.complete('a')
    // Its only branch is its unconditional continuation, the gate: completing
    // the boundary advances to it rather than leaving the cursor on the fork.
    bare.complete('fork')
    expect(ids(bare)).toEqual(['gate'])
    bare.verdict('gate', 'reloop')
    expect(ids(bare)).toEqual(['a'])

    // A gate that is itself a branch of a fork whose reloop target is outside
    // the fork body is refused at load time (cross-container reloop).
    expect(() => new Spine({
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', writes: [], command: 'true' },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['c1', 'gate'] },
        c1: { _type: 'FnNode', id: 'c1', writes: [], command: 'true' },
        join: { _type: 'JoinNode', id: 'join', sources: ['c1', 'gate'] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: 'true' },
      },
      edges: [
        { source: 'a', target: 'fork' },
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'end', condition: 'proceed' },
      ],
    })).toThrow(/gate gate is inside container fork.*reloop target a is outside/)
  })

  it('rewinds through a node two branches share without visiting it twice', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', writes: [], command: 'true' },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['b', 'c'] },
        b: { _type: 'FnNode', id: 'b', writes: [], command: 'true' },
        c: { _type: 'FnNode', id: 'c', writes: [], command: 'true' },
        shared: { _type: 'FnNode', id: 'shared', writes: [], command: 'true' },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: 'true' },
      },
      edges: [
        { source: 'a', target: 'fork' },
        { source: 'b', target: 'shared' },
        { source: 'c', target: 'shared' },
        { source: 'shared', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'end', condition: 'proceed' },
      ],
    })
    spine.complete('a')
    spine.complete('fork')
    spine.complete('b')
    // Both branches name `shared`; it is scheduled once, not twice.
    expect(ids(spine)).toEqual(['c', 'shared'])
    spine.complete('c')
    expect(ids(spine)).toEqual(['shared'])
    spine.complete('shared')
    // Both branches name `shared`; the rewind has to reach it once and stop at the gate.
    spine.verdict('gate', 'reloop')
    expect(ids(spine)).toEqual(['a'])
    expect(spine.state().gate_verdicts).toEqual({ gate: 'reloop' })
  })
})

describe('refusing what the cursor cannot do', () => {
  const graph: WorkflowGraph = {
    name: 't',
    start_node: 'gate',
    nodes: {
      gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] },
      on: { _type: 'FnNode', id: 'on', writes: [], command: 'true' },
    },
    edges: [{ source: 'gate', target: 'on', condition: 'proceed' }],
  }

  it('refuses a node the graph does not have, for both reads and completion', () => {
    const spine = new Spine(graph)
    expect(() => spine.view('ghost')).toThrow(/unknown node ghost/)
    expect(() => spine.complete('ghost')).toThrow(/unknown node ghost/)
  })

  it('refuses to complete a gate, which only a verdict may route', () => {
    const spine = new Spine(graph)
    expect(() => spine.complete('gate')).toThrow(/is a gate — use verdict\(\)/)
  })

  it('refuses a verdict for a node that is not a gate, and an empty verdict', () => {
    const spine = new Spine(graph)
    expect(() => spine.verdict('on', 'proceed')).toThrow(/on is not a gate/)
    expect(() => spine.verdict('gate', '   ')).toThrow(/empty verdict for gate gate/)
  })

  it('does not record a skipped node as complete', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'a',
      nodes: { a: { _type: 'FnNode', id: 'a', writes: [], command: 'true' } },
      edges: [],
    })
    spine.skip('a')
    // A skipped node leaves the cursor, so completing it is refused rather than
    // recorded; either way it is never marked complete.
    expect(() => spine.complete('a')).toThrow(/node a is not due/)
    expect(spine.state().completed).toEqual([])
  })

  it('marks a node type the spine cannot execute as unsupported', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'exotic',
      nodes: { exotic: { _type: 'SubgraphForkNode', id: 'exotic' } },
      edges: [],
    })
    expect(spine.view('exotic').unsupported).toBe(true)
  })
})

describe('incomplete cursor drain', () => {
  it('halts naming unreached sinks when the cursor empties short of the end', () => {
    // Restore a checkpoint whose cursor is empty short of the end.
    // Graph: a → b, where b is the sink. Complete a, then restore a
    // checkpoint with cursor=[] and b not completed.
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    // Save a checkpoint at cursor=['b'], then tamper to empty cursor before restore.
    const saved = JSON.parse(JSON.stringify(spine.checkpoint()))
    saved.state.cursor = []
    saved.state.completed = ['a']
    const second = new Spine(graph)
    second.restore(saved)
    const result = second.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeDefined()
      expect(result.halted!.reason).toBe('incomplete')
      expect(result.halted!.unreached).toEqual(['b'])
      // gate field is empty for incomplete — node names live in unreached
      expect(result.halted!.gate).toBe('')
    }
  })

  it('finishes done with no halt when one of two alternative sinks is reached', () => {
    // Graph: gate → b (proceed), gate → c (alt). Both b and c are sinks.
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'gate',
      nodes: {
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'FnNode', id: 'b', writes: [], command: '' },
        c: { _type: 'FnNode', id: 'c', writes: [], command: '' },
      },
      edges: [
        { source: 'gate', target: 'b', condition: 'proceed' },
        { source: 'gate', target: 'c', condition: 'alt' },
      ],
    }
    const spine = new Spine(graph)
    spine.verdict('gate', 'proceed')
    spine.complete('b')
    const result = spine.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeUndefined()
    }
  })

  it('does not treat fork branch targets as unreached sinks', () => {
    // Graph: fork → join → end. c1, c2 are fork targets with no outgoing edges.
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['c1', 'c2'] },
        c1: { _type: 'AgentNode', id: 'c1', role: 'reviewer', prompt_template: 'r1', writes: [], reads: [] },
        c2: { _type: 'AgentNode', id: 'c2', role: 'reviewer', prompt_template: 'r2', writes: [], reads: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['c1', 'c2'] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: '' },
      },
      edges: [
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'end' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('fork')
    spine.complete('c1')
    spine.complete('c2')
    spine.complete('join')
    spine.complete('end')
    const result = spine.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeUndefined()
    }
  })

  it('skipping the sink counts as reached — done with no halt', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'FnNode', id: 'b', writes: [], command: '' },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    spine.skip('b')
    const result = spine.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeUndefined()
    }
  })

  it('data body exit is not a graph sink — halts as incomplete naming finish', () => {
    // Graph shaped like dataset_probe: items → body → exit (same node), items → finish
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'items',
      nodes: {
        items: { _type: 'DataNode', id: 'items', task: 'probe', subgraph_entry: 'work', subgraph_exit: 'work' },
        work: { _type: 'FnNode', id: 'work', writes: [], command: '' },
        finish: { _type: 'FnNode', id: 'finish', writes: [], command: '' },
      },
      edges: [{ source: 'items', target: 'finish' }],
    }
    const spine = new Spine(graph)
    spine.startPass('items', 1)
    spine.complete('work')
    spine.endPass()
    // Now cursor is at ['finish']. Save checkpoint and tamper to simulate lost continuation.
    const saved = JSON.parse(JSON.stringify(spine.checkpoint()))
    saved.state.cursor = []
    saved.state.completed = ['items', 'work']
    delete saved.state.pass
    const second = new Spine(graph)
    second.restore(saved)
    const result = second.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeDefined()
      expect(result.halted!.reason).toBe('incomplete')
      expect(result.halted!.unreached).toEqual(['finish'])
    }
  })

  it('data graph run to completion finishes done', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'items',
      nodes: {
        items: { _type: 'DataNode', id: 'items', task: 'probe', subgraph_entry: 'work', subgraph_exit: 'work' },
        work: { _type: 'FnNode', id: 'work', writes: [], command: '' },
        finish: { _type: 'FnNode', id: 'finish', writes: [], command: '' },
      },
      edges: [{ source: 'items', target: 'finish' }],
    }
    const spine = new Spine(graph)
    spine.startPass('items', 1)
    spine.complete('work')
    spine.endPass()
    spine.complete('finish')
    const result = spine.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeUndefined()
    }
  })

  it('fork graph incomplete drain still names only the real sink', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['c1', 'c2'] },
        c1: { _type: 'AgentNode', id: 'c1', role: 'reviewer', prompt_template: 'r1', writes: [], reads: [] },
        c2: { _type: 'AgentNode', id: 'c2', role: 'reviewer', prompt_template: 'r2', writes: [], reads: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['c1', 'c2'] },
        end: { _type: 'FnNode', id: 'end', writes: [], command: '' },
      },
      edges: [
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'end' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('fork')
    spine.complete('c1')
    spine.complete('c2')
    spine.complete('join')
    // Save checkpoint with join completed but end not reached
    const saved = JSON.parse(JSON.stringify(spine.checkpoint()))
    saved.state.cursor = []
    saved.state.completed = ['fork', 'c1', 'c2', 'join']
    const second = new Spine(graph)
    second.restore(saved)
    const result = second.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeDefined()
      expect(result.halted!.reason).toBe('incomplete')
      expect(result.halted!.unreached).toEqual(['end'])
    }
  })

  it('director graph incomplete drain names the successor, not expanded children', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'dir',
      nodes: {
        dir: {
          _type: 'AgentNode', id: 'dir', role: 'worker', prompt_template: 'D', writes: [], reads: [],
          expand: { plan: 'p.json', role: 'worker', slug_field: 'slug', prompt_field: 'prompt', writes_template: '' },
        },
        after: { _type: 'FnNode', id: 'after', writes: [], command: '' },
      },
      edges: [{ source: 'dir', target: 'after' }],
    }
    const spine = new Spine(graph)
    spine.expand('dir', [{ slug: 'x', prompt: 'p1' }, { slug: 'y', prompt: 'p2' }])
    spine.complete('dir__x')
    spine.complete('dir__y')
    // After completing both children, cursor drains, afters pops 'after'
    // Save checkpoint and tamper to simulate incomplete drain
    const saved = JSON.parse(JSON.stringify(spine.checkpoint()))
    saved.state.cursor = []
    saved.state.completed = ['dir', 'dir__x', 'dir__y']
    const second = new Spine(graph)
    second.restore(saved)
    const result = second.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeDefined()
      expect(result.halted!.reason).toBe('incomplete')
      expect(result.halted!.unreached).toEqual(['after'])
    }
  })

  it('graph with data node and several alternative ends', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'items',
      nodes: {
        items: { _type: 'DataNode', id: 'items', task: 'probe', subgraph_entry: 'work', subgraph_exit: 'work' },
        work: { _type: 'FnNode', id: 'work', writes: [], command: '' },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end_a: { _type: 'FnNode', id: 'end_a', writes: [], command: '' },
        end_b: { _type: 'FnNode', id: 'end_b', writes: [], command: '' },
      },
      edges: [
        { source: 'items', target: 'gate' },
        { source: 'gate', target: 'end_a', condition: 'proceed' },
        { source: 'gate', target: 'end_b', condition: 'alt' },
      ],
    }
    const spine = new Spine(graph)
    spine.startPass('items', 1)
    spine.complete('work')
    spine.endPass()
    spine.verdict('gate', 'proceed')
    spine.complete('end_a')
    const result = spine.next()
    expect(result.status).toBe('done')
    if (result.status === 'done') {
      expect(result.halted).toBeUndefined()
    }
  })
})

describe('gate refusal reason persistence', () => {
  // Common graph: emit → check → done_node.
  // check is a GateNode with max_iterations: 1, reloop edge back to emit, proceed edge to done_node.
  const gateReasonGraph: WorkflowGraph = {
    name: 't',
    start_node: 'emit',
    nodes: {
      emit: { _type: 'AgentNode', id: 'emit', role: 'builder', prompt_template: 'E', writes: [], reads: [] },
      check: { _type: 'GateNode', id: 'check', evaluator_type: 'fn', evaluator_command: 'true', reads: [], max_iterations: 1 },
      done_node: { _type: 'FnNode', id: 'done_node', writes: [], command: '' },
    },
    edges: [
      { source: 'emit', target: 'check' },
      { source: 'check', target: 'emit', condition: 'reloop' },
      { source: 'check', target: 'done_node', condition: 'proceed' },
    ],
  }

  it('rerun without text shows the gate reason after halt', () => {
    const spine = new Spine(gateReasonGraph)
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'fix the formatting')
    // count=1, cap=1: 1 > 1 is false → loops
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'still wrong')
    // count=2, cap=1: 2 > 1 → halts
    const haltResult = spine.next()
    expect(haltResult.status).toBe('done')
    if (haltResult.status === 'done') {
      expect(haltResult.halted?.reason).toBe('max_iterations')
    }
    // Verify gateReasons in state
    const reasons = (spine.state().gate_reasons as Record<string, { reason: string }>)
    expect(reasons.check?.reason).toBe('still wrong')
    // Redo emit
    spine.redo('emit')
    // gateReasonFor should return 'still wrong'
    expect(spine.gateReasonFor('emit')).toBe('still wrong')
    // Compose feedback as index.ts does (no rerun text)
    const gateReason = spine.gateReasonFor('emit')
    if (gateReason !== undefined) spine.setFeedback('emit', gateReason)
    expect(spine.next().feedback?.emit).toBe('still wrong')
  })

  it('rerun with text shows reason first, then text', () => {
    const spine = new Spine(gateReasonGraph)
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'fix the formatting')
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'still wrong')
    spine.redo('emit')
    // Compose as index.ts does: gate reason + rerun text
    const gateReason = spine.gateReasonFor('emit')
    const rerunText = 'try a different approach'
    const parts = [gateReason, rerunText].filter(Boolean)
    spine.setFeedback('emit', parts.join('\n\n'))
    const fb = spine.next().feedback?.emit ?? ''
    expect(fb).toMatch(/^still wrong/)
    expect(fb).toContain('try a different approach')
    // Reason comes before rerun text
    expect(fb.indexOf('still wrong')).toBeLessThan(fb.indexOf('try a different approach'))
  })

  it('rerun after the gate passed shows no old reason', () => {
    const spine = new Spine(gateReasonGraph)
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'fix it')
    spine.complete('emit')
    spine.verdict('check', 'proceed')
    spine.complete('done_node')
    // Redo emit after gate proceeded
    spine.redo('emit')
    expect(spine.gateReasonFor('emit')).toBeUndefined()
    expect(spine.next().feedback).not.toHaveProperty('emit')
  })

  it('with multiple gates, gives the latest refusal', () => {
    // Graph: emit → check (reloop→emit) → gate_qa (reloop→emit) → done_node
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'emit',
      nodes: {
        emit: { _type: 'AgentNode', id: 'emit', role: 'builder', prompt_template: 'E', writes: [], reads: [] },
        check: { _type: 'GateNode', id: 'check', evaluator_type: 'fn', evaluator_command: 'true', reads: [], max_iterations: 3 },
        gate_qa: { _type: 'GateNode', id: 'gate_qa', evaluator_type: 'fn', evaluator_command: 'true', reads: [], max_iterations: 3 },
        done_node: { _type: 'FnNode', id: 'done_node', writes: [], command: '' },
      },
      edges: [
        { source: 'emit', target: 'check' },
        { source: 'check', target: 'emit', condition: 'reloop' },
        { source: 'check', target: 'gate_qa', condition: 'proceed' },
        { source: 'gate_qa', target: 'emit', condition: 'reloop' },
        { source: 'gate_qa', target: 'done_node', condition: 'proceed' },
      ],
    }
    const spine = new Spine(graph)
    // First pass: check reloops
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'check says fix it')
    // Second pass: check proceeds, gate_qa reloops
    spine.complete('emit')
    spine.verdict('check', 'proceed')
    spine.verdict('gate_qa', 'reloop', 'qa says fix it')
    // Both gates hold reasons for emit. gate_qa's is later.
    expect(spine.gateReasonFor('emit')).toBe('qa says fix it')
  })

  it('gate reason survives checkpoint round-trip', () => {
    const spine = new Spine(gateReasonGraph)
    spine.complete('emit')
    spine.verdict('check', 'reloop', 'fix it')
    const saved = JSON.parse(JSON.stringify(spine.checkpoint()))
    const second = new Spine(gateReasonGraph)
    second.restore(saved)
    expect(second.gateReasonFor('emit')).toBe('fix it')
  })
})

describe('a fork with nothing to fan out to', () => {
  it('ends the run when the fork names no branches and no successor', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'fork',
      nodes: { fork: { _type: 'ForkNode', id: 'fork' } },
      edges: [],
    }
    expect(new Spine(graph).complete('fork').status).toBe('done')
    expect(new Spine(graph).skip('fork').status).toBe('done')
  })

  it('skip() with provenance opts records person source', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.skip('b', { source: 'person', at: '2026-01-01T00:00:00Z' })
    const state = spine.state()
    expect(state.skip_provenance).toEqual({ b: { source: 'person', at: '2026-01-01T00:00:00Z' } })
    expect(state.skipped).toEqual(['b'])
  })

  type SkipProvenance = Record<string, { source: string; at: string }>;

  function prov(state: Record<string, unknown>): SkipProvenance {
    return (state.skip_provenance ?? {}) as SkipProvenance
  }

  it('skip() without opts defaults to ceo source', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.skip('b')
    const state = spine.state()
    expect(prov(state)).toBeDefined()
    expect(prov(state).b!.source).toBe('ceo')
    expect(typeof prov(state).b!.at).toBe('string')
  })

  it('planSkip with person source bypasses conditional-gate refusal', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'b', condition: 'proceed' },
        { source: 'gate', target: 'a', condition: 'reloop' },
      ],
    }
    const spine = new Spine(graph)
    // CEO skip should throw for conditional-only gate
    expect(() => spine.planSkip('gate')).toThrow(/conditional successors/)
    // Person skip should succeed
    expect(() => spine.planSkip('gate', { source: 'person' })).not.toThrow()
  })

  it('unskip clears provenance', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.skip('b', { source: 'person', at: '2026-01-01T00:00:00Z' })
    expect(prov(spine.state()).b).toBeDefined()
    spine.unskip('b')
    expect(prov(spine.state()).b).toBeUndefined()
  })

  it('invalidateNodes clears provenance when clearSkips is true', () => {
    // Graph: a → gate (reloop→a, proceed→b), b
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'b', condition: 'proceed' },
        { source: 'gate', target: 'a', condition: 'reloop' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    // gate is now in cursor; skip it with person provenance.
    // The routing rule in next() routes proceed for the person-skipped gate.
    spine.skip('gate', { source: 'person', at: '2026-01-01T00:00:00Z' })
    // Redo the gate's upstream node (a), which calls invalidateNodes
    // with clearSkips=true for all descendants (gate, b).
    // planRedo refuses gates directly; redo a instead.
    spine.redo('a')
    const state = spine.state()
    // After redo, the gate's skip_provenance should be cleared
    expect(prov(state).gate).toBeUndefined()
  })

  it('restore reads skip_provenance from checkpoint', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.skip('b', { source: 'person', at: '2026-01-01T00:00:00Z' })
    const checkpoint = spine.checkpoint()
    const spine2 = new Spine(graph)
    spine2.restore(checkpoint)
    expect(spine2.state().skip_provenance).toEqual({ b: { source: 'person', at: '2026-01-01T00:00:00Z' } })
  })

  it('restore tolerates missing skip_provenance', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.skip('b')
    const checkpoint = spine.checkpoint()
    // Remove skip_provenance from saved data
    delete (checkpoint.state as any).skip_provenance
    const spine2 = new Spine(graph)
    spine2.restore(checkpoint)
    expect(spine2.state().skip_provenance).toEqual({})
  })

  it('provenance survives unskip and re-skip', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.skip('b', { source: 'person', at: '2026-01-01T00:00:00Z' })
    spine.unskip('b')
    expect(prov(spine.state()).b).toBeUndefined()
    spine.skip('b', { source: 'person', at: '2026-02-01T00:00:00Z' })
    expect(prov(spine.state()).b).toEqual({ source: 'person', at: '2026-02-01T00:00:00Z' })
  })

  it('person-skip routing rule routes conditional-only gate via proceed', () => {
    // Graph: a → gate (all conditional edges), gate proceed → b, gate reloop → a
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', gate_prompt: 'ok?', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'b', condition: 'proceed' },
        { source: 'gate', target: 'a', condition: 'reloop' },
      ],
    }
    // Case A: skip queued before check is due
    const spineA = new Spine(graph)
    spineA.skip('gate', { source: 'person', at: '2026-01-01T00:00:00Z' })
    // gate is not in cursor yet (a is), so routing hasn't fired
    spineA.complete('a')
    // Now gate enters cursor and the routing rule should fire
    const nextA = spineA.next()
    expect(nextA.status).toBe('ready')
    expect(nextA.nodes!.map((n: any) => n.id)).toEqual(['b'])

    // Case B: skip when gate is due
    const spineB = new Spine(graph)
    spineB.complete('a')
    // gate is now in cursor
    spineB.skip('gate', { source: 'person', at: '2026-01-01T00:00:00Z' })
    const nextB = spineB.next()
    expect(nextB.status).toBe('ready')
    expect(nextB.nodes!.map((n: any) => n.id)).toEqual(['b'])
  })

  it('person-skip survives checkpoint/restore (case D)', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', gate_prompt: 'ok?', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'b', condition: 'proceed' },
        { source: 'gate', target: 'a', condition: 'reloop' },
      ],
    }
    const spine = new Spine(graph)
    spine.skip('gate', { source: 'person', at: '2026-01-01T00:00:00Z' })
    const checkpoint = spine.checkpoint()
    const spine2 = new Spine(graph)
    spine2.restore(checkpoint)
    spine2.complete('a')
    const next = spine2.next()
    expect(next.status).toBe('ready')
    expect(next.nodes!.map((n: any) => n.id)).toEqual(['b'])
  })

  it('is inert when its targets are already skipped, and ends the run without a join', () => {
    const spine = new Spine({
      name: 't',
      start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['c1'] },
        c1: { _type: 'FnNode', id: 'c1', writes: [], command: 'true' },
      },
      edges: [],
    })
    spine.skip('c1')
    expect(spine.skip('fork').status).toBe('done')
    expect(() => spine.skip('fork')).not.toThrow()
    expect(spine.state().skipped).toEqual(['c1', 'fork'])
  })
})
