import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { Spine, validateGraph } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

function ids(spine: Spine): string[] {
  const next = spine.next()
  return next.nodes?.map(node => node.id) ?? []
}

/** A switch gate: two forward outcomes, neither of them a rewind. */
const switchGraph: WorkflowGraph = {
  name: 'switch',
  start_node: 'decide',
  nodes: {
    decide: { _type: 'GateNode', id: 'decide', evaluator_type: 'fn', reads: [] },
    draft: { _type: 'FnNode', id: 'draft', command: 'echo draft', reads: [], writes: [] },
    debug: { _type: 'FnNode', id: 'debug', command: 'echo debug', reads: [], writes: [] },
    join: { _type: 'FnNode', id: 'join', command: 'true', reads: [], writes: [] },
    end: { _type: 'FnNode', id: 'end', command: 'true', reads: [], writes: [] },
  },
  edges: [
    { source: 'decide', target: 'draft', condition: 'DRAFT' },
    { source: 'decide', target: 'debug', condition: 'DEBUG' },
    { source: 'draft', target: 'join' },
    { source: 'debug', target: 'join' },
    { source: 'join', target: 'end' },
  ],
}

describe('named gate outcomes', () => {
  it('routes to a gate-named forward branch, matching case-insensitively', () => {
    const spine = new Spine(switchGraph)
    expect(ids(spine)).toEqual(['decide'])
    spine.verdict('decide', 'DRAFT')
    expect(ids(spine)).toEqual(['draft'])
    spine.complete('draft')
    expect(ids(spine)).toEqual(['join'])
  })

  it('lists the gate outcomes it accepts', () => {
    const spine = new Spine(switchGraph)
    expect(spine.next().nodes?.[0]?.verdicts).toEqual(['draft', 'debug'])
  })

  it('rejects an outcome the gate does not declare, naming the accepted ones', () => {
    const spine = new Spine(switchGraph)
    expect(() => spine.verdict('decide', 'improve')).toThrow(/accepts draft, debug/)
  })

  it('still treats reloop as a rewind, not a forward branch', () => {
    const spine = new Spine(switchGraph)
    expect(() => spine.verdict('decide', 'reloop')).toThrow(/no reloop edge/)
  })
})

describe('gate-declared reloop cap', () => {
  const graph: WorkflowGraph = {
    name: 'capped',
    start_node: 'work',
    nodes: {
      work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
      stop: { _type: 'GateNode', id: 'stop', evaluator_type: 'fn', reads: [], max_iterations: 5 },
      exit: { _type: 'FnNode', id: 'exit', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'work', target: 'stop' },
      { source: 'stop', target: 'work', condition: 'reloop' },
      { source: 'stop', target: 'exit', condition: 'proceed' },
    ],
  }

  it('uses the gate cap rather than the runner default', () => {
    const spine = new Spine(graph, { maxIterations: 1 })
    for (let index = 0; index < 5; index += 1) {
      spine.complete('work')
      spine.verdict('stop', 'reloop')
      expect(spine.next().status).toBe('ready')
    }
    spine.complete('work')
    spine.verdict('stop', 'reloop')
    expect(spine.next().status).toBe('done')
  })

  it('exposes the cap on the gate view', () => {
    const spine = new Spine(graph)
    spine.complete('work')
    expect(spine.next().nodes?.[0]?.max_iterations).toBe(5)
  })
})

describe('multi-node parallel branches', () => {
  const graph: WorkflowGraph = {
    name: 'fan',
    start_node: 'fork',
    nodes: {
      fork: { _type: 'ForkNode', id: 'fork', targets: ['a1', 'b1'] },
      a1: { _type: 'FnNode', id: 'a1', command: 'true', reads: [], writes: [] },
      a2: { _type: 'FnNode', id: 'a2', command: 'true', reads: [], writes: [] },
      b1: { _type: 'FnNode', id: 'b1', command: 'true', reads: [], writes: [] },
      b2: { _type: 'FnNode', id: 'b2', command: 'true', reads: [], writes: [] },
      join: { _type: 'JoinNode', id: 'join', sources: ['a2', 'b2'] },
      end: { _type: 'FnNode', id: 'end', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'fork', target: 'join' },
      { source: 'a1', target: 'a2' },
      { source: 'b1', target: 'b2' },
      { source: 'join', target: 'end' },
    ],
  }

  it('advances each branch independently and barriers at the join', () => {
    const spine = new Spine(graph)
    spine.complete('fork')
    expect(ids(spine)).toEqual(['a1', 'b1'])
    spine.complete('a1')
    expect(ids(spine)).toEqual(['b1', 'a2'])
    spine.complete('a2')
    expect(ids(spine)).toEqual(['b1'])
    spine.complete('b1')
    expect(ids(spine)).toEqual(['b2'])
    spine.complete('b2')
    expect(ids(spine)).toEqual(['join'])
    spine.complete('join')
    expect(ids(spine)).toEqual(['end'])
  })
})

describe('validateGraph', () => {
  it('accepts a walkable graph', () => {
    expect(validateGraph(switchGraph)).toEqual([])
  })

  it('reports a missing start node and dangling edges', () => {
    const broken: WorkflowGraph = {
      name: 'broken',
      start_node: 'nope',
      nodes: { a: { _type: 'FnNode', id: 'a', command: 'true' } },
      edges: [{ source: 'a', target: 'ghost' }, { source: 'ghost', target: 'a' }],
    }
    const messages = validateGraph(broken).map(issue => issue.message)
    expect(messages).toContain("start_node 'nope' is not a node")
    expect(messages).toContain("edge target 'ghost' is not a node")
    expect(messages).toContain("edge source 'ghost' is not a node")
  })

  it('reports a gate with no outgoing edge, and a fork or join that names a node that is not one', () => {
    const broken: WorkflowGraph = {
      name: 'broken',
      start_node: 'gate',
      nodes: {
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user' },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['ghost'] },
        join: { _type: 'JoinNode', id: 'join', sources: ['ghost'] },
      },
      edges: [],
    }
    const messages = validateGraph(broken).map(issue => issue.message)
    expect(messages).toContain('gate has no outgoing edge')
    expect(messages).toContain("fork target 'ghost' is not a node")
    expect(messages).toContain("join source 'ghost' is not a node")
  })

  it('accepts a fork and a join that declare no branches', () => {
    expect(validateGraph({
      name: 't',
      start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork' },
        join: { _type: 'JoinNode', id: 'join' },
      },
      edges: [{ source: 'fork', target: 'join' }],
    })).toEqual([])
  })

  it('follows a fork\'s targets when deciding what the walk can reach', () => {
    expect(validateGraph({
      name: 't',
      start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['c1'] },
        c1: { _type: 'FnNode', id: 'c1', command: 'true', reads: [], writes: [] },
        stranded: { _type: 'FnNode', id: 'stranded', command: 'true', reads: [], writes: [] },
      },
      edges: [],
    })).toEqual([{ node: 'stranded', message: 'unreachable from start_node' }])
  })

  it('reports a node with two unconditional edges, which has no defined successor', () => {
    const ambiguous: WorkflowGraph = {
      name: 'ambiguous',
      start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', command: 'true' },
        b: { _type: 'FnNode', id: 'b', command: 'true' },
        c: { _type: 'FnNode', id: 'c', command: 'true' },
      },
      edges: [{ source: 'a', target: 'b' }, { source: 'a', target: 'c' }],
    }
    expect(validateGraph(ambiguous).map(issue => issue.message)).toEqual([
      'node has more than one unconditional edge',
    ])
  })

  it('reports a duplicated gate verdict and a non-executable node type', () => {
    const duplicated: WorkflowGraph = {
      name: 'duplicated',
      start_node: 'gate',
      nodes: {
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn' },
        a: { _type: 'FnNode', id: 'a', command: 'true' },
        b: { _type: 'FnNode', id: 'b', command: 'true' },
        exotic: { _type: 'SubgraphForkNode', id: 'exotic' },
      },
      edges: [
        { source: 'gate', target: 'a', condition: 'proceed' },
        { source: 'gate', target: 'b', condition: 'PROCEED' },
        { source: 'gate', target: 'exotic' },
      ],
    }
    const messages = validateGraph(duplicated).map(issue => issue.message)
    expect(messages).toContain("two edges carry the verdict 'proceed'")
    expect(messages).toContain("node type 'SubgraphForkNode' is not executable")
  })

  it('reports nodes the walk can never reach', () => {
    const orphaned: WorkflowGraph = {
      name: 'orphaned',
      start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', command: 'true' },
        b: { _type: 'FnNode', id: 'b', command: 'true' },
        stranded: { _type: 'FnNode', id: 'stranded', command: 'true' },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    expect(validateGraph(orphaned)).toEqual([
      { node: 'stranded', message: 'unreachable from start_node' },
    ])
  })

  it('accepts DataNode and SelectionNode as executable vocabulary', () => {
    const graph: WorkflowGraph = {
      name: 'data-driven',
      start_node: 'data',
      nodes: {
        data: {
          _type: 'DataNode', id: 'data',
          inline_items: [{ id: 'i1' }, { id: 'i2' }],
          subgraph_entry: 'work', subgraph_exit: 'end',
          reads: [], writes: [],
        },
        work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
        end: { _type: 'FnNode', id: 'end', command: 'true', reads: [], writes: [] },
        after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
        pick: { _type: 'SelectionNode', id: 'pick', strategy: 'best_score', reads: [], writes: [] },
      },
      edges: [
        { source: 'data', target: 'after' },
        { source: 'work', target: 'end' },
        { source: 'after', target: 'pick' },
      ],
    }
    expect(validateGraph(graph)).toEqual([])
    const spine = new Spine(graph, { maxIterations: 2 })
    // The data node is not handed over: its first item's pass is.
    expect(spine.next().nodes?.[0]).toMatchObject({ id: 'data', _type: 'DataNode' })
    expect(spine.startPass('data', 2).nodes?.[0]).toMatchObject({ id: 'work', _type: 'FnNode' })
  })

  it('describes a data node to a reader without handing it over to be iterated', () => {
    const graph: WorkflowGraph = {
      name: 'data-view',
      start_node: 'data',
      nodes: {
        data: {
          _type: 'DataNode', id: 'data',
          source_path: '/data', source_format: 'jsonl',
          subgraph_entry: 'a', subgraph_exit: 'b',
          limit: 7, reads: [], writes: [],
        },
        a: { _type: 'FnNode', id: 'a', command: 'true', reads: [], writes: [] },
        b: { _type: 'FnNode', id: 'b', command: 'true', reads: [], writes: [] },
        after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
      },
      edges: [
        { source: 'data', target: 'after' },
        { source: 'a', target: 'b' },
      ],
    }
    expect(validateGraph(graph)).toEqual([])
    const spine = new Spine(graph, { maxIterations: 2 })
    expect(spine.view('data')).toMatchObject({
      id: 'data',
      source_path: '/data',
      source_format: 'jsonl',
      subgraph_entry: 'a',
      subgraph_exit: 'b',
      limit: 7,
    })
    expect(spine.startPass('data', 1).nodes?.[0]).toMatchObject({ id: 'a' })
  })
})

describe('describing nodes that declare only their type', () => {
  const spine = new Spine({
    name: 'sparse',
    start_node: 'gate',
    nodes: {
      gate: { _type: 'GateNode', id: 'gate' },
      agent: { _type: 'AgentNode', id: 'agent' },
      llm: { _type: 'LLMNode', id: 'llm' },
      instance: { _type: 'LLMNode', id: 'instance', instance_prompt: 'per-instance' },
      fork: { _type: 'ForkNode', id: 'fork' },
      join: { _type: 'JoinNode', id: 'join' },
      fn: { _type: 'FnNode', id: 'fn' },
      data: { _type: 'DataNode', id: 'data' },
      pick: { _type: 'SelectionNode', id: 'pick' },
    },
    edges: [],
  })

  it('leaves a gate bare rather than inventing an evaluator or labels', () => {
    const view = spine.view('gate')
    expect(view.evaluator_type).toBeUndefined()
    expect(view.gate_prompt).toBe('')
    expect(view.reads).toEqual([])
    expect(view.verdicts).toEqual([])
  })

  it('describes a bare agent node with an empty prompt and no files', () => {
    const view = spine.view('agent')
    expect(view.writes).toEqual([])
    expect(view.reads).toEqual([])
    expect(view.prompt_template).toBe('')
  })

  it('falls back to the instance prompt, and then to nothing, for a bare LLM node', () => {
    const bare = spine.view('llm')
    expect(bare.writes).toEqual([])
    expect(bare.reads).toEqual([])
    expect(bare.prompt_template).toBe('')
    expect(bare.model).toBe('')
    expect(bare.temperature).toBe(0)
    expect(spine.view('instance').prompt_template).toBe('per-instance')
  })

  it('describes a branch-less fork and join as empty', () => {
    expect(spine.view('fork').targets).toEqual([])
    expect(spine.view('join').sources).toEqual([])
  })

  it('describes a bare fn node with no writes and no command', () => {
    const view = spine.view('fn')
    expect(view.writes).toEqual([])
    expect(view.command).toBe('')
  })

  it('describes a bare data node with empty source and subgraph fields', () => {
    const view = spine.view('data')
    expect(view.writes).toEqual([])
    expect(view.reads).toEqual([])
    expect(view.subgraph_entry).toBe('')
    expect(view.subgraph_exit).toBe('')
  })

  it('defaults a selection node to the best_score strategy', () => {
    const view = spine.view('pick')
    expect(view.strategy).toBe('best_score')
    expect(view.reads).toEqual([])
    expect(view.writes).toEqual([])
  })
})

describe('an op\'s knob_outputs', () => {
  /** prepare as Python lowers it: the bare question carries no knob, compare's prompt carries its task. */
  function withPrepare(fields: Record<string, unknown>): WorkflowGraph {
    return {
      name: 'knob-outputs',
      start_node: 'prepare',
      nodes: {
        prepare: {
          _type: 'FnNode', id: 'prepare', command: 'true', reads: ['current_item.json'], writes: [],
          knobs: ['compare_task'], produces: ['user_prompt', 'compare_prompt'],
          knob_outputs: { user_prompt: [], compare_prompt: ['compare_task'] },
          ...fields,
        },
      },
      edges: [],
    }
  }

  it('accepts a declaration that names every output with the op\'s own knobs', () => {
    expect(validateGraph(withPrepare({}))).toEqual([])
  })

  it.each([
    [{ user_prompt: [], compare_prompt: ['compare_task'], ghost: [] }, 'knob_outputs names ghost, which the node neither writes nor produces'],
    [{ compare_prompt: ['compare_task'] }, 'knob_outputs leaves out user_prompt; it names every output'],
    [{ user_prompt: ['skill'], compare_prompt: ['compare_task'] }, "knob_outputs['user_prompt'] names skill, which the node does not read"],
    [{ user_prompt: [], compare_prompt: ['compare_task', 'compare_task'] }, "knob_outputs['compare_prompt'] names a knob twice"],
    [{ user_prompt: 'compare_task', compare_prompt: [] }, "knob_outputs['user_prompt'] is a list of knob names"],
    [['user_prompt'], 'knob_outputs is a mapping of each output to the knobs it carries'],
  ])('refuses %j', (declared, message) => {
    expect(validateGraph(withPrepare({ knob_outputs: declared })))
      .toEqual([{ node: 'prepare', message }])
  })

  it.each([
    ['livemath-skill_graph', 'prepare', { user_prompt: [], compare_prompt: ['compare_task'] }],
    ['livemath-solver', 'format_prompt', { user_prompt: [], solver_prompt: [], skill_section: ['skill'] }],
  ])('accepts %s as Python lowers it, with %s\'s declaration', (name, op, declared) => {
    const path = fileURLToPath(new URL(`../../../../refactory/tests/lowering_snapshots/${name}.graph.json`, import.meta.url))
    const graph = JSON.parse(readFileSync(path, 'utf8')) as WorkflowGraph
    expect(graph.nodes[op]?.knob_outputs).toEqual(declared)
    expect(validateGraph(graph)).toEqual([])
  })

  const shared = JSON.parse(readFileSync(fileURLToPath(new URL(
    '../../../../refactory/tests/fixtures/knob_outputs_cases.json', import.meta.url)), 'utf8')) as {
    op: { knobs: string[], writes: string[], produces: string[] }
    cases: Array<{ knob_outputs: unknown, why: string | null }>
  }
  it.each(shared.cases)('reads the declarations the walker reads the same way: %j', ({ knob_outputs, why }) => {
    const graph: WorkflowGraph = {
      name: 'shared', start_node: 'build', edges: [],
      nodes: { build: { _type: 'FnNode', id: 'build', command: 'true', reads: [], ...shared.op, knob_outputs } as WorkflowGraph['nodes'][string] },
    }
    const messages = validateGraph(graph).map(issue => issue.message)
    if (why === null) expect(messages).toEqual([])
    else expect(messages.some(message => message.includes(why))).toBe(true)
  })

  it('refuses it on anything but an op', () => {
    const graph = withPrepare({})
    const prepare = graph.nodes.prepare
    if (prepare === undefined) throw new Error('withPrepare builds a prepare node')
    graph.nodes.prepare = { ...prepare, _type: 'AgentNode' }
    expect(validateGraph(graph)).toContainEqual(
      { node: 'prepare', message: "node type 'AgentNode' declares knob_outputs; only an op does" })
  })
})
