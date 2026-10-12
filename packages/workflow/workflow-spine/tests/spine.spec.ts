import { describe, expect, it } from 'vitest'
import { Spine, validateGraph } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

function ids(spine: Spine): string[] {
  const next = spine.next()
  return next.nodes?.map(node => node.id) ?? []
}

it('preserves a native model gate stage timeout in the dispatched view', () => {
  const spine = new Spine({ name: 'candidate', start_node: 'ceo_gate', nodes: {
    ceo_gate: { id: 'ceo_gate', _type: 'GateNode', evaluator_type: 'agent', command_timeout_s: 3600,
      evaluator_command: 'python -m refactory_dsh.run refactory_dsh.create_fidelity_gates:evaluate_native' },
    done: { id: 'done', _type: 'FnNode', command: 'true' },
  }, edges: [{ source: 'ceo_gate', target: 'done', condition: 'proceed' }] })
  expect(spine.next().nodes?.[0]).toMatchObject({ evaluator_type: 'agent', command_timeout_s: 3600 })
})

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

    // The gate is itself a branch of the fork whose rewind passes it.
    const branching = new Spine({
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
    })
    branching.complete('a')
    branching.complete('fork')
    branching.complete('c1')
    expect(ids(branching)).toEqual(['gate'])
    branching.verdict('gate', 'reloop')
    expect(ids(branching)).toEqual(['a'])
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
