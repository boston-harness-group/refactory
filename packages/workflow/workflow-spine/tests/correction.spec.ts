/**
 * Correcting a run in flight: reopening a completed node with `redo`, amending an
 * agent/llm node's prompt with `amendPrompt`, and the `wf_redo` / `wf_edit` tools
 * that carry a person's correction from the CEO to the spine.
 *
 * The spine specs drive the state machine directly; the tool specs mount the
 * plugin the way `composition.spec.ts` does, because a correction only counts
 * once it reaches the spine through the tool boundary. Each test builds its own
 * graph: the spine holds the node bodies it was given, so an amendment mutates
 * the graph object itself.
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { afterEach, describe, expect, it } from 'vitest'
import * as spinePlugin from '../src/index.ts'
import { Spine } from '../src/spine.ts'
import type { Config } from '../src/index.ts'
import type { WorkflowGraph } from '../src/types.ts'

const scratchDirs: string[] = []

afterEach(() => {
  for (const dir of scratchDirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-spine-correction-'))
  scratchDirs.push(dir)
  return dir
}

function ids(spine: Spine): string[] {
  const next = spine.next()
  return next.nodes?.map(node => node.id) ?? []
}

/** A graph with every node kind `redo` has an opinion about. */
function redoGraph(): WorkflowGraph {
  return {
    name: 't',
    start_node: 'a',
    nodes: {
      a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
      b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
      data: { _type: 'DataNode', id: 'data', subgraph_entry: 'sub', subgraph_exit: 'subend', reads: [], writes: [] },
      sub: { _type: 'AgentNode', id: 'sub', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
      subend: { _type: 'AgentNode', id: 'subend', role: 'builder', prompt_template: 'SE', reads: [], writes: [] },
    },
    edges: [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'gate' },
      { source: 'gate', target: 'data', condition: 'proceed' },
      { source: 'gate', target: 'a', condition: 'reloop' },
      { source: 'data', target: 'sub' },
      { source: 'sub', target: 'subend' },
    ],
  }
}

describe('redo', () => {
  it('reopens a completed node, moving the cursor back to it and clearing completion', () => {
    const spine = new Spine(redoGraph())
    spine.complete('a')
    expect(ids(spine)).toEqual(['b'])
    const next = spine.redo('a')
    expect(next.status).toBe('ready')
    expect(ids(spine)).toEqual(['a'])
    expect(spine.state().completed).not.toContain('a')
    // The log records the redo as a line naming where it went and what it invalidated.
    expect(spine.state().log).toEqual(expect.arrayContaining([
      expect.stringMatching(/^redid a at \S+; invalidated \S/),
    ]))
  })

  it('clears a skip as well as the completion', () => {
    const spine = new Spine(redoGraph())
    spine.complete('a')
    // A completed node cannot be skipped now, so the skipped work is a due
    // descendant; reopening `a` must clear both histories it invalidates.
    spine.skip('b')
    expect(spine.state().skipped).toContain('b')
    spine.redo('a')
    expect(ids(spine)).toEqual(['a'])
    expect(spine.state().skipped).not.toContain('b')
    expect(spine.state().completed).not.toContain('a')
  })

  it('refuses an unknown node', () => {
    expect(() => new Spine(redoGraph()).redo('ghost')).toThrow(/unknown node ghost/)
  })

  it('refuses a node that has not completed', () => {
    expect(() => new Spine(redoGraph()).redo('b'))
      .toThrow(/node b has not completed, so there is nothing to redo/)
  })

  it('refuses a gate, naming the upstream agent to replay', () => {
    expect(() => new Spine(redoGraph()).redo('gate'))
      .toThrow(/gate is a gate; replay b instead/)
  })

  it('refuses a data node, whose passes are the runner\'s', () => {
    expect(() => new Spine(redoGraph()).redo('data'))
      .toThrow(/node data iterates its items; redo a completed node in its body or upstream/)
  })
})

/** A graph with the prompt-bearing node kinds and one that carries no prompt. */
function amendGraph(): WorkflowGraph {
  return {
    name: 't',
    start_node: 'a',
    nodes: {
      a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
      blank: { _type: 'AgentNode', id: 'blank', role: 'builder', reads: [], writes: [] },
      llm: { _type: 'LLMNode', id: 'llm', system_prompt: 'S', model: 'flash', reads: [], writes: [] },
      alt: { _type: 'LLMNode', id: 'alt', instance_prompt: 'I', model: 'flash', reads: [], writes: [] },
      fn: { _type: 'FnNode', id: 'fn', command: 'true', reads: [], writes: [] },
    },
    edges: [],
  }
}

describe('amendPrompt', () => {
  it('appends to an existing prompt on its own paragraph, and the view carries it', () => {
    const spine = new Spine(amendGraph())
    spine.amendPrompt('a', 'extra', 'append')
    expect(spine.view('a').prompt_template).toBe('A\n\nextra')
    expect(spine.state().prompts).toEqual({ a: 'A\n\nextra' })
    expect(spine.state().log).toContain('prompt a append')
  })

  it('sets a prompt in place of the existing one', () => {
    const spine = new Spine(amendGraph())
    spine.amendPrompt('a', 'replaced', 'set')
    expect(spine.view('a').prompt_template).toBe('replaced')
    expect(spine.state().prompts).toEqual({ a: 'replaced' })
  })

  it('sets when there is no existing prompt', () => {
    const spine = new Spine(amendGraph())
    spine.amendPrompt('blank', 'first', 'append')
    expect(spine.view('blank').prompt_template).toBe('first')
  })

  it('reports only the nodes whose prompt was amended', () => {
    const spine = new Spine(amendGraph())
    spine.amendPrompt('a', 'x', 'set')
    expect(spine.state().prompts).toEqual({ a: 'x' })
    expect(spine.state().prompts).not.toHaveProperty('blank')
  })

  it('amends an LLM node\'s system prompt, which is what its view reports', () => {
    const spine = new Spine(amendGraph())
    spine.amendPrompt('llm', 'more', 'append')
    expect(spine.view('llm').prompt_template).toBe('S\n\nmore')
    expect(spine.state().prompts).toEqual({ llm: 'S\n\nmore' })
  })

  it('amends an LLM node that declared only an instance prompt', () => {
    const spine = new Spine(amendGraph())
    spine.amendPrompt('alt', 'more', 'append')
    expect(spine.view('alt').prompt_template).toBe('I\n\nmore')
  })

  it('refuses an unknown node', () => {
    expect(() => new Spine(amendGraph()).amendPrompt('ghost', 'x', 'set')).toThrow(/unknown node ghost/)
  })

  it('refuses a node type that carries no prompt, naming the type', () => {
    expect(() => new Spine(amendGraph()).amendPrompt('fn', 'x', 'set'))
      .toThrow(/node fn is a FnNode and carries no prompt/)
  })

  it('refuses empty text', () => {
    expect(() => new Spine(amendGraph()).amendPrompt('a', '   ', 'append'))
      .toThrow('prompt amendment text is required')
  })

  it('refuses a set-prompt that drops a placeholder', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder',
             prompt_template: 'Build {{name}}.\n{{emit_scope}}\nDo the work.',
             reads: [], writes: [] },
      },
      edges: [],
    }
    const spine = new Spine(graph)
    expect(() => spine.amendPrompt('a', 'just fix it', 'set'))
      .toThrow(/set-prompt refused.*emit_scope.*append-prompt/)
  })

  it('allows a set-prompt that keeps all placeholders', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder',
             prompt_template: 'Build {{name}}.\n{{emit_scope}}\nDo the work.',
             reads: [], writes: [] },
      },
      edges: [],
    }
    const spine = new Spine(graph)
    spine.amendPrompt('a', 'Build {{name}} better.\n{{emit_scope}}\nDo it right.', 'set')
    expect(spine.view('a').prompt_template).toBe('Build {{name}} better.\n{{emit_scope}}\nDo it right.')
  })

  it('allows append-prompt on a template with placeholders', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder',
             prompt_template: 'Build {{name}}.\n{{emit_scope}}\nDo the work.',
             reads: [], writes: [] },
      },
      edges: [],
    }
    const spine = new Spine(graph)
    spine.amendPrompt('a', 'also handle errors', 'append')
    expect(spine.view('a').prompt_template).toContain('{{name}}')
    expect(spine.view('a').prompt_template).toContain('{{emit_scope}}')
    expect(spine.view('a').prompt_template).toContain('also handle errors')
  })

  it('does not guard a template whose only placeholders are in the steering block', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder',
             prompt_template: '<!-- refactory steering context -->\n{{brief}}\n<!-- end refactory steering context -->\nDo the work.',
             reads: [], writes: [] },
      },
      edges: [],
    }
    const spine = new Spine(graph)
    spine.amendPrompt('a', 'fix it', 'set')
    expect(spine.view('a').prompt_template).toBe('fix it')
  })

  it('amends the prompt of an AgentNode that has a dispatch command', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: {
          _type: 'AgentNode', id: 'a', role: 'builder',
          prompt_template: 'original task',
          dispatch_command: 'python -m worker build',
          reads: [], writes: [],
        },
      },
      edges: [],
    }
    const spine = new Spine(graph)
    spine.amendPrompt('a', 'also handle edge cases', 'append')
    expect(spine.view('a').prompt_template).toBe('original task\n\nalso handle edge cases')
    expect(spine.view('a').command).toBe('python -m worker build')
    expect(spine.state().prompts).toEqual({ a: 'original task\n\nalso handle edge cases' })
  })
})

/** One agent node, so the cursor has something a model must do. */
function agentGraph(): WorkflowGraph {
  return {
    name: 't',
    start_node: 'a',
    nodes: { a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] } },
    edges: [],
  }
}

function config(graph: WorkflowGraph = agentGraph(), overrides: Partial<Config> = {}): Config {
  return {
    graph,
    toolPrefix: 'wf',
    maxIterations: 3,
    interactiveOpen: false,
    openCommand: '',
    openArgs: [],
    executeCommands: false,
    commandEnv: {},
    commandTimeoutMs: 10_000,
    commandSandboxMode: 'danger-full-access',
    dataCommand: '',
    steerCommand: '',
    steerArgs: [],
    ceoDeny: [],
    ...overrides,
  }
}

/** A context with the two services the plugin injects, and the plugin mounted. */
async function mount(graph: WorkflowGraph = agentGraph(), overrides: Partial<Config> = {}): Promise<Context> {
  const root = new Context()
  await root.plugin(SystemPrompt)
  await root.plugin(ToolRuntime)
  await root.plugin(spinePlugin, config(graph, overrides))
  return root
}

async function call(ctx: Context, name: string, args: unknown = {}): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} not registered`)
  return definition.execute(args, {} as never)
}

/** A graph with a reloop from gate back to a. */
function reloopGraph(): WorkflowGraph {
  return {
    name: 't',
    start_node: 'a',
    nodes: {
      a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
      b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
      c: { _type: 'AgentNode', id: 'c', role: 'builder', prompt_template: 'C', reads: [], writes: [] },
    },
    edges: [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'gate' },
      { source: 'gate', target: 'a', condition: 'reloop' },
      { source: 'gate', target: 'c', condition: 'proceed' },
    ],
  }
}

describe('forward-only invalidation', () => {
  it('redo of a node before a gate with reloop invalidates only forward descendants', () => {
    const spine = new Spine(reloopGraph())
    spine.complete('a')
    spine.complete('b')
    spine.verdict('gate', 'proceed')
    spine.complete('c')
    const plan = spine.planRedo('a')
    const nodes = new Set(plan.nodes)
    expect(nodes).toContain('a')
    expect(nodes).toContain('b')
    expect(nodes).toContain('gate')
    expect(nodes).toContain('c')
    // No extra nodes reachable only through reloop back-edge.
    expect(plan.nodes.length).toBe(4)
  })

  it('redo of overwatch-pattern graph excludes loop body reached via back-edge', () => {
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'emit',
      nodes: {
        emit: { _type: 'AgentNode', id: 'emit', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
        materialise: { _type: 'FnNode', id: 'materialise', command: 'true', reads: [], writes: [] },
        check: { _type: 'FnNode', id: 'check', command: 'true', reads: [], writes: [] },
        fork_qa: { _type: 'ForkNode', id: 'fork_qa', targets: ['review_a', 'review_b', 'review_c'] },
        review_a: { _type: 'AgentNode', id: 'review_a', role: 'reviewer', prompt_template: 'RA', reads: [], writes: [] },
        review_b: { _type: 'AgentNode', id: 'review_b', role: 'reviewer', prompt_template: 'RB', reads: [], writes: [] },
        review_c: { _type: 'AgentNode', id: 'review_c', role: 'reviewer', prompt_template: 'RC', reads: [], writes: [] },
        join_qa: { _type: 'JoinNode', id: 'join_qa', sources: ['review_a', 'review_b', 'review_c'] },
        synth: { _type: 'AgentNode', id: 'synth', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        gate_qa: { _type: 'GateNode', id: 'gate_qa', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        overwatch: { _type: 'AgentNode', id: 'overwatch', role: 'reviewer', prompt_template: 'O', reads: [], writes: [] },
        gate_overwatch: { _type: 'GateNode', id: 'gate_overwatch', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        report: { _type: 'AgentNode', id: 'report', role: 'builder', prompt_template: 'R', reads: [], writes: [] },
        deliver: { _type: 'AgentNode', id: 'deliver', role: 'builder', prompt_template: 'D', reads: [], writes: [] },
      },
      edges: [
        { source: 'emit', target: 'materialise' },
        { source: 'materialise', target: 'check' },
        { source: 'check', target: 'fork_qa' },
        { source: 'fork_qa', target: 'join_qa' },
        { source: 'join_qa', target: 'synth' },
        { source: 'synth', target: 'gate_qa' },
        { source: 'gate_qa', target: 'emit', condition: 'reloop' },
        { source: 'gate_qa', target: 'overwatch', condition: 'proceed' },
        { source: 'overwatch', target: 'gate_overwatch' },
        { source: 'gate_overwatch', target: 'emit', condition: 'reloop' },
        { source: 'gate_overwatch', target: 'report', condition: 'proceed' },
        { source: 'report', target: 'deliver' },
      ],
    }
    const spine = new Spine(graph)
    // Complete all through the proceed path
    spine.complete('emit')
    spine.complete('materialise')
    spine.complete('check')
    spine.complete('fork_qa')
    spine.complete('review_a')
    spine.complete('review_b')
    spine.complete('review_c')
    spine.complete('join_qa')
    spine.complete('synth')
    spine.verdict('gate_qa', 'proceed')
    spine.complete('overwatch')
    spine.verdict('gate_overwatch', 'proceed')
    spine.complete('report')
    spine.complete('deliver')

    const plan = spine.planRedo('overwatch')
    const nodes = new Set(plan.nodes)
    expect(nodes).toContain('overwatch')
    expect(nodes).toContain('gate_overwatch')
    expect(nodes).toContain('report')
    expect(nodes).toContain('deliver')
    // Must NOT contain loop body nodes
    expect(nodes).not.toContain('emit')
    expect(nodes).not.toContain('materialise')
    expect(nodes).not.toContain('check')
    expect(nodes).not.toContain('fork_qa')
    expect(nodes).not.toContain('review_a')
    expect(nodes).not.toContain('review_b')
    expect(nodes).not.toContain('review_c')
    expect(nodes).not.toContain('join_qa')
    expect(nodes).not.toContain('synth')
    expect(nodes).not.toContain('gate_qa')
  })

  it('node reachable via both forward and back paths is still invalidated', () => {
    // a → x → gate → (reloop→a, proceed→c)
    // x is forward-reachable from a
    const graph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        x: { _type: 'AgentNode', id: 'x', role: 'builder', prompt_template: 'X', reads: [], writes: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        c: { _type: 'AgentNode', id: 'c', role: 'builder', prompt_template: 'C', reads: [], writes: [] },
      },
      edges: [
        { source: 'a', target: 'x' },
        { source: 'x', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'c', condition: 'proceed' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    spine.complete('x')
    spine.verdict('gate', 'proceed')
    spine.complete('c')
    const plan = spine.planRedo('a')
    expect(new Set(plan.nodes)).toContain('x')
  })
})

describe('setFeedback', () => {
  it('sets per-visit feedback that appears in next()', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.setFeedback('a', 'fix this')
    const next = spine.next()
    expect(next.feedback?.a).toBe('fix this')
  })

  it('rerun text as feedback is scoped to one visit and cleared on complete', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    spine.redo('a')
    spine.setFeedback('a', 'one-shot fix')
    expect(spine.next().feedback?.a).toBe('one-shot fix')
    spine.complete('a')
    expect(spine.next().feedback).not.toHaveProperty('a')
    // Redo again — the one-shot text is gone
    spine.redo('a')
    expect(spine.next().feedback).not.toHaveProperty('a')
  })

  it('gate reloop replaces rerun feedback with gate findings', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'b', condition: 'proceed' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    spine.redo('a')
    spine.setFeedback('a', 'one-shot')
    spine.complete('a')
    spine.verdict('gate', 'reloop', 'gate findings')
    expect(spine.state().feedback).toHaveProperty('a', 'gate findings')
  })

  it('refuses an unknown node', () => {
    const spine = new Spine(reloopGraph())
    expect(() => spine.setFeedback('ghost', 'x')).toThrow('unknown node ghost')
  })

  it('rerun feedback survives checkpoint restore', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    spine.complete('a')
    spine.redo('a')
    spine.setFeedback('a', 'fix')
    const saved = JSON.parse(JSON.stringify(spine.checkpoint()))
    const restored = new Spine(graph)
    restored.restore(saved)
    expect(restored.state().feedback).toHaveProperty('a', 'fix')
    restored.complete('a')
    expect(restored.state().feedback).not.toHaveProperty('a')
  })
})

describe('gate refusal message', () => {
  it('names the upstream agent in the error', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'b', condition: 'proceed' },
      ],
    }
    const spine = new Spine(graph)
    expect(() => spine.planRedo('gate')).toThrow(/gate is a gate; replay a instead/)
  })
})

describe('deferred redo', () => {
  function forkGraph(): WorkflowGraph {
    return {
      name: 't', start_node: 'start',
      nodes: {
        start: { _type: 'AgentNode', id: 'start', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['p', 'q'] },
        p: { _type: 'AgentNode', id: 'p', role: 'builder', prompt_template: 'P', reads: [], writes: [] },
        q: { _type: 'AgentNode', id: 'q', role: 'builder', prompt_template: 'Q', reads: [], writes: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['p', 'q'] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'start', target: 'fork' },
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'end' },
      ],
    }
  }

  it('planRedo with deferred flag returns plan during concurrent work', () => {
    const spine = new Spine(forkGraph())
    spine.complete('start')
    spine.complete('fork')
    spine.complete('p')
    // q is still due, concurrent work running
    expect(spine.hasConcurrentWork()).toBe(true)
    expect(() => spine.planRedo('p')).toThrow(/concurrent branches/)
    const plan = spine.planRedo('p', { deferred: true })
    expect(plan.nodes).toContain('p')
  })

  it('checkpoint includes deferred_redo_plans for concurrency-blocked nodes', () => {
    const spine = new Spine(forkGraph())
    spine.complete('start')
    spine.complete('fork')
    spine.complete('p')
    const cp = spine.checkpoint()
    expect(cp.deferred_redo_plans).toHaveProperty('p')
    expect(cp.redo_plans).not.toHaveProperty('p')
  })

  it('hasConcurrentWork returns false when no fork is active', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const spine = new Spine(graph)
    expect(spine.hasConcurrentWork()).toBe(false)
  })
})

describe('the correction tools', () => {
  it('registers wf_redo', async () => {
    expect((await mount()).tools.get('wf_redo')).toBeDefined()
  })

  it('reopens a completed node so wf_next returns it again, and it leaves completed', async () => {
    const ctx = await mount()
    await call(ctx, 'wf_done', { node: 'a' })
    expect(await call(ctx, 'wf_next')).toMatchObject({ status: 'done' })

    await call(ctx, 'wf_redo', { node: 'a' })
    expect(await call(ctx, 'wf_next')).toMatchObject({ status: 'ready', nodes: [{ id: 'a' }] })
    expect((await call(ctx, 'wf_state') as { completed: string[] }).completed).not.toContain('a')
  })

  it('sets a node\'s prompt through wf_edit, and the next view carries it', async () => {
    const ctx = await mount()
    await call(ctx, 'wf_edit', { node: 'a', action: 'set-prompt', text: 'corrected' })
    expect(await call(ctx, 'wf_next')).toMatchObject({ nodes: [{ id: 'a', prompt_template: 'corrected' }] })
    expect(await call(ctx, 'wf_state')).toMatchObject({ prompts: { a: 'corrected' } })
  })

  it('appends through wf_edit', async () => {
    const ctx = await mount()
    await call(ctx, 'wf_edit', { node: 'a', action: 'append-prompt', text: 'more' })
    expect(await call(ctx, 'wf_next')).toMatchObject({ nodes: [{ prompt_template: 'A\n\nmore' }] })
  })

  it('amends and redoes a dispatched AgentNode through the tools', async () => {
    const dispatched: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: {
          _type: 'AgentNode', id: 'a', role: 'builder',
          prompt_template: 'build it',
          dispatch_command: 'echo dispatched',
          reads: [], writes: [],
        },
      },
      edges: [],
    }
    const ctx = await mount(dispatched)
    // The dispatched node is non-deterministic here (executeCommands: false),
    // so it appears in wf_next for the CEO. But we can still edit its prompt.
    await call(ctx, 'wf_edit', { node: 'a', action: 'append-prompt', text: 'also handle errors' })
    const next = await call(ctx, 'wf_next') as { nodes: Array<{ prompt_template: string }> }
    expect(next.nodes[0]!.prompt_template).toBe('build it\n\nalso handle errors')

    // Complete and redo: the amended prompt survives.
    await call(ctx, 'wf_done', { node: 'a' })
    await call(ctx, 'wf_redo', { node: 'a' })
    const redone = await call(ctx, 'wf_next') as { nodes: Array<{ prompt_template: string }> }
    expect(redone.nodes[0]!.prompt_template).toBe('build it\n\nalso handle errors')
  })

  it('refuses a prompt action with no text', async () => {
    const ctx = await mount()
    await expect(call(ctx, 'wf_edit', { node: 'a', action: 'set-prompt' }))
      .rejects.toThrow(/needs a non-empty prompt amendment/)
  })

  it('still skips and unskips through wf_edit', async () => {
    const ctx = await mount()
    await call(ctx, 'wf_edit', { node: 'a', action: 'skip' })
    expect(await call(ctx, 'wf_state')).toMatchObject({ skipped: ['a'] })
    await call(ctx, 'wf_edit', { node: 'a', action: 'unskip' })
    expect(await call(ctx, 'wf_state')).toMatchObject({ skipped: [] })
  })
})

describe('applyControls integration', () => {
  /** Write a rerun control file into the workdir. */
  function writeRerunControl(workdir: string, target: string, opts: {
    revision: string, text?: string, deferredNodes?: string[]
  }): void {
    const directory = join(workdir, '.refactory', 'graph-controls')
    mkdirSync(directory, { recursive: true })
    const control: Record<string, unknown> = {
      id: '1', action: 'rerun', target, text: opts.text ?? '',
      revision: opts.revision, run_id: resolve(workdir), invocation: null,
      status: 'queued',
    }
    if (opts.deferredNodes) control.deferred_nodes = opts.deferredNodes
    writeFileSync(join(directory, '1.json'), JSON.stringify(control))
  }

  it('uses setFeedback not amendPrompt for rerun text', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir })
    // wf_next writes the first checkpoint.
    await call(ctx, 'wf_next')
    // Complete 'a'.
    await call(ctx, 'wf_done', { node: 'a' })
    // wf_next writes the checkpoint with the post-completion state.
    await call(ctx, 'wf_next')
    // Read the checkpoint's graph_revision.
    const revision = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    // Queue a rerun control with text.
    writeRerunControl(workdir, 'a', { revision, text: 'one-shot fix' })
    // The next wf_next triggers applyControls at its first boundary.
    const next = await call(ctx, 'wf_next') as { status: string; nodes: Array<{ id: string; prompt_template: string }>; feedback: Record<string, string> }
    // The node is reopened, prompt is unchanged, and feedback carries the text.
    expect(next.nodes.find(n => n.id === 'a')?.prompt_template).toBe('A')
    expect(next.feedback?.a).toBe('one-shot fix')
    // The control file was consumed.
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(false)
  })

  it('feedback sidecar is written by drain and reaches the dispatched node manifest (dm-110)', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [],
             dispatch_command: 'echo dispatched' },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    // First wf_next dispatches 'a' and writes the prompt file.
    const executed: Array<{ command: string; env: Record<string, string> }> = []
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string; env?: Record<string, string> }) => {
        executed.push({ command: spec.command, env: spec.env ?? {} })
        return { result: async () => ({
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text: '', truncated: false }, stderr: { text: '', truncated: false },
        }) }
      },
    })
    await call(ctx, 'wf_next')
    // Node 'a' is auto-completed by the shell (dispatch_command exits with code 0).
    // Read checkpoint's graph_revision (written after 'a' completed).
    const revision = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    // Queue a rerun control with text.
    writeRerunControl(workdir, 'a', { revision, text: 'MARKER-dm110-fix-this' })
    // The next wf_next triggers applyControls which sets feedback, then drain writes the sidecar.
    executed.length = 0
    await call(ctx, 'wf_next')
    // The feedback sidecar should exist.
    const fbFile = join(workdir, '.refactory', 'prompts', 'a.feedback.md')
    expect(existsSync(fbFile)).toBe(true)
    expect(readFileSync(fbFile, 'utf8')).toContain('MARKER-dm110-fix-this')
    // The manifest in the environment should reference the feedback file.
    const lastExec = executed[executed.length - 1]
    expect(lastExec).toBeDefined()
    const manifest = JSON.parse(lastExec!.env?.DSH_WORKFLOW_NODE ?? '{}')
    expect(manifest.feedback_file).toBe(fbFile)
  })

  it('gate reloop reason reaches the reloop target feedback sidecar (dm-110)', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [],
             dispatch_command: 'echo dispatched' },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn',
                evaluator_command: 'echo gate-eval', reads: [] },
        c: { _type: 'AgentNode', id: 'c', role: 'builder', prompt_template: 'C', reads: [], writes: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'c', condition: 'proceed' },
      ],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    let callCount = 0
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string }) => {
        callCount += 1
        // First two calls: 'a' runs then gate evaluates.
        // Gate returns "GATE-REASON\nreloop" the first time.
        if (spec.command.includes('gate-eval')) {
          return { result: async () => ({
            exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
            stdout: { text: 'GATE-REASON-TEXT\nreloop', truncated: false },
            stderr: { text: '', truncated: false },
          }) }
        }
        return { result: async () => ({
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text: '', truncated: false }, stderr: { text: '', truncated: false },
        }) }
      },
    })
    // First wf_next: runs 'a', then runs gate which reloops back to 'a'.
    await call(ctx, 'wf_next')
    // After the reloop, 'a' should be dispatched again with the gate reason as feedback.
    const fbFile = join(workdir, '.refactory', 'prompts', 'a.feedback.md')
    expect(existsSync(fbFile)).toBe(true)
    expect(readFileSync(fbFile, 'utf8')).toContain('GATE-REASON-TEXT')
  })

  it('drain dispatches to real worker_node.execute and feedback reaches prompt (dm-110 e2e)', async () => {
    // Cross-boundary test: the TypeScript spine's drain() writes the feedback
    // sidecar and manifest, a real shell runs the Python dispatch command, and
    // the real worker_node.execute() reads the sidecar.  This is the test that
    // catches a format mismatch between drain() and worker_node.py — neither
    // side tested alone can.
    const workdir = scratch()

    // Write the worker spec the dispatch helper will hand to execute().
    const specsDir = join(workdir, '.refactory', 'worker-specs')
    mkdirSync(specsDir, { recursive: true })
    const specPath = join(specsDir, 'a-spec.json')
    writeFileSync(specPath, JSON.stringify({
      schema_version: 1, node_id: 'a', prompt: 'Build it.',
      reads: [], writes: [],
      model_context: { provider: 'test', model: 'test' },
    }))

    // Write the Python dispatch helper.  It calls the real worker_node.execute()
    // with a scripted model and records the prompt to a file the test reads.
    const helperPath = join(workdir, 'dispatch_helper.py')
    writeFileSync(helperPath, [
      '#!/usr/bin/env python3',
      '"""Cross-boundary dm-110 test helper: dispatch command that wraps',
      'worker_node.execute() with a scripted model recording the prompt."""',
      'import json, os, sys',
      'from pathlib import Path',
      'from unittest import mock',
      '',
      'spec_path = sys.argv[1]',
      'spec = json.loads(Path(spec_path).read_text())',
      'workdir = Path.cwd()',
      'manifest = json.loads(os.environ.get("DSH_WORKFLOW_NODE", "{}"))',
      'node_id = manifest.get("id", "unknown")',
      '',
      'counter_file = workdir / ".refactory" / f"dm110-visit-{node_id}.txt"',
      'counter_file.parent.mkdir(parents=True, exist_ok=True)',
      'v = int(counter_file.read_text()) + 1 if counter_file.exists() else 1',
      'counter_file.write_text(str(v))',
      '',
      'captured = []',
      '',
      'class M:',
      '    def __init__(self, **k): pass',
      '    def for_node(self, n, role="worker"): return self',
      '    def complete(self, p, workdir=None):',
      '        captured.append(p); return "done"',
      '    def context(self): return {"provider": "test", "model": "test"}',
      '    def route(self): return {"worker_backend": "dsh"}',
      '',
      'class D:',
      '    @staticmethod',
      '    def begin(w, n, p, ws, **k):',
      '        captured.append(p); return {"prompt": p, "node": n, "status": "pending"}',
      '    @staticmethod',
      '    def finish(w, n, **k): return {"node": n, "status": "delivered"}',
      '    @staticmethod',
      '    def _path(w, name): return Path(w) / name',
      '',
      'with mock.patch("refactory_dsh.model.DshModelClient", M), \\',
      '     mock.patch("refactory_dsh.agent_delivery.begin", D.begin), \\',
      '     mock.patch("refactory_dsh.agent_delivery.finish", D.finish), \\',
      '     mock.patch("refactory_dsh.agent_delivery._path", D._path), \\',
      '     mock.patch("refactory_dsh.run._knobs", return_value={}), \\',
      '     mock.patch("refactory_dsh.run._read_facts", return_value={}), \\',
      '     mock.patch("refactory_dsh.nn.executor.render", side_effect=lambda p, c: p), \\',
      '     mock.patch("refactory_dsh.nn.executor.RunContext"):',
      '    from refactory_dsh import worker_node',
      '    worker_node.execute(spec, workdir)',
      '',
      'out = workdir / ".refactory" / f"dm110-prompt-{v}-{node_id}.txt"',
      'out.write_text(json.dumps(captured))',
    ].join('\n'))

    const pythonBin = join(process.cwd(), '.venv', 'bin', 'python')
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'Build it.',
             reads: [], writes: [],
             dispatch_command: `${pythonBin} ${helperPath} ${specPath}` },
      },
      edges: [],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })

    // A shell that really runs the command, so worker_node.py runs for real.
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 30_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string> }) => {
        const started = spawnSync('/bin/sh', ['-c', spec.command], {
          cwd: spec.workdir, encoding: 'utf8', timeout: spec.timeoutMs,
          env: { ...process.env, ...(spec.env ?? {}) },
        })
        return { result: async () => ({
          exitCode: started.status ?? 0, signal: started.signal,
          timedOut: false, aborted: false, timeoutMs: spec.timeoutMs ?? 30_000,
          stdout: { text: started.stdout ?? '', truncated: false },
          stderr: { text: started.stderr ?? '', truncated: false },
        }) }
      },
    })

    // Visit 1: dispatch without feedback.
    await call(ctx, 'wf_next')
    const v1File = join(workdir, '.refactory', 'dm110-prompt-1-a.txt')
    expect(existsSync(v1File)).toBe(true)
    const v1 = JSON.parse(readFileSync(v1File, 'utf8')) as string[]
    expect(v1.some(p => p.includes('Feedback for this visit'))).toBe(false)
    expect(v1.some(p => p.includes('Build it.'))).toBe(true)

    // Get checkpoint graph_revision and queue a rerun with text.
    const revision = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    writeRerunControl(workdir, 'a', { revision, text: 'MARKER-dm110-cross-boundary-fix' })

    // Visit 2: dispatch with feedback.
    await call(ctx, 'wf_next')
    const v2File = join(workdir, '.refactory', 'dm110-prompt-2-a.txt')
    expect(existsSync(v2File)).toBe(true)
    const v2 = JSON.parse(readFileSync(v2File, 'utf8')) as string[]
    expect(v2.some(p => p.includes('## Feedback for this visit'))).toBe(true)
    expect(v2.some(p => p.includes('MARKER-dm110-cross-boundary-fix'))).toBe(true)
    expect(v2.some(p => p.includes('Build it.'))).toBe(true)
  })

  it('feedback sidecar is not written when no feedback exists, so stale files are not read (dm-110)', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [],
             dispatch_command: 'echo dispatched' },
      },
      edges: [],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async () => ({
        result: async () => ({
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text: '', truncated: false }, stderr: { text: '', truncated: false },
        }),
      }),
    })
    // First wf_next: runs 'a' with no feedback — no sidecar should exist.
    await call(ctx, 'wf_next')
    const fbFile = join(workdir, '.refactory', 'prompts', 'a.feedback.md')
    expect(existsSync(fbFile)).toBe(false)
  })

  it('deferred rerun control survives concurrent work and applies after join', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'start',
      nodes: {
        start: { _type: 'AgentNode', id: 'start', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['p', 'q'] },
        p: { _type: 'AgentNode', id: 'p', role: 'builder', prompt_template: 'P', reads: [], writes: [] },
        q: { _type: 'AgentNode', id: 'q', role: 'builder', prompt_template: 'Q', reads: [], writes: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['p', 'q'] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'start', target: 'fork' },
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'end' },
      ],
    }
    const ctx = await mount(graph, { workdir })
    // wf_next writes checkpoint, returns 'start' as due.
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'start' })
    // wf_next returns fork as due (executeCommands is false).
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'fork' })
    // wf_next returns p and q as due concurrently.
    await call(ctx, 'wf_next')
    // Complete p; q still pending.
    await call(ctx, 'wf_done', { node: 'p' })
    // wf_next writes a checkpoint with p completed and q still pending.
    await call(ctx, 'wf_next')
    // Read checkpoint to get graph_revision and deferred plan.
    const cp = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8'))
    const revision = cp.graph_revision
    const deferredNodes = cp.deferred_redo_plans?.p?.nodes as string[]
    expect(deferredNodes).toBeDefined()
    // Queue a deferred rerun control for 'p'.
    writeRerunControl(workdir, 'p', { revision, deferredNodes })
    // wf_next — q is still due with join in afters, so hasConcurrentWork() is
    // true and applyControls defers the control.
    await call(ctx, 'wf_next')
    // The control file should still exist (deferred, not consumed).
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(true)
    // Complete q — the join becomes sole cursor, afters empty.
    await call(ctx, 'wf_done', { node: 'q' })
    // wf_next — concurrent work finished (cursor=['join'], afters=[]).
    // report() calls applyControls which applies the deferred control:
    // hasConcurrentWork() returns false, redo('p') reopens p, cursor → ['p'].
    const result = await call(ctx, 'wf_next') as { status: string; nodes: Array<{ id: string }> }
    // The control file should have been consumed.
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(false)
    // The receipt should say applied.
    const receipt = JSON.parse(readFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), 'utf8'))
    expect(receipt.status).toBe('applied')
    // Node 'p' should be back in due nodes (uncompleted).
    expect(result.nodes.some(n => n.id === 'p')).toBe(true)
    const state = await call(ctx, 'wf_state') as { completed: string[] }
    expect(state.completed).not.toContain('p')
  })

  it('deferred rerun applies after three branches complete at separate boundaries', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'start',
      nodes: {
        start: { _type: 'AgentNode', id: 'start', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['p', 'q', 'r'] },
        p: { _type: 'AgentNode', id: 'p', role: 'builder', prompt_template: 'P', reads: [], writes: [] },
        q: { _type: 'AgentNode', id: 'q', role: 'builder', prompt_template: 'Q', reads: [], writes: [] },
        r: { _type: 'AgentNode', id: 'r', role: 'builder', prompt_template: 'R', reads: [], writes: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['p', 'q', 'r'] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'start', target: 'fork' },
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'end' },
      ],
    }
    const ctx = await mount(graph, { workdir })
    // 1. wf_next writes checkpoint, returns 'start' as due.
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'start' })
    // 2. wf_next returns fork as due.
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'fork' })
    // 3. wf_next returns p, q, r as due concurrently.
    await call(ctx, 'wf_next')
    // 4. Complete p; q and r still pending.
    await call(ctx, 'wf_done', { node: 'p' })
    // 5. wf_next writes checkpoint with p completed, q and r still due.
    await call(ctx, 'wf_next')
    // 6. Read checkpoint to get graph_revision and deferred plan.
    const cp = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8'))
    const revision = cp.graph_revision
    const deferredNodes = cp.deferred_redo_plans?.p?.nodes as string[]
    expect(deferredNodes).toBeDefined()
    // 7. Queue a deferred rerun control for 'p' with this revision.
    writeRerunControl(workdir, 'p', { revision, deferredNodes })
    // 8. wf_next — q and r still due, concurrent work running, control deferred.
    await call(ctx, 'wf_next')
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(true)
    // 9. Complete q.
    await call(ctx, 'wf_done', { node: 'q' })
    // 10. wf_next — checkpoint rewritten (q completed), r still due, control deferred.
    await call(ctx, 'wf_next')
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(true)
    // 11. Capture the graph_revision after q finished. It must differ from
    //     the original revision, proving the revision check would have rejected.
    const revisionAfterQ = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    expect(revisionAfterQ).not.toBe(revision)
    // 12. Complete r.
    await call(ctx, 'wf_done', { node: 'r' })
    // 13. wf_next — join drains, concurrent work ends. applyControls applies the
    //     deferred control: plan check passes, revision check skipped, redo('p').
    const result = await call(ctx, 'wf_next') as { status: string; nodes: Array<{ id: string }> }
    // 14. Assertions.
    // The control file was consumed.
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(false)
    // The receipt says applied.
    const receipt = JSON.parse(readFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), 'utf8'))
    expect(receipt.status).toBe('applied')
    // Node 'p' is back in the due nodes.
    expect(result.nodes.some(n => n.id === 'p')).toBe(true)
    // p is no longer completed.
    const state = await call(ctx, 'wf_state') as { completed: string[] }
    expect(state.completed).not.toContain('p')
  })

  it('wf_redo refuses when run is halted by max_iterations', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'b', condition: 'proceed' },
      ],
    }
    const ctx = await mount(graph, { workdir, maxIterations: 2 })
    // Drive through 3 reloops (count > cap=2 halts on the 3rd)
    await call(ctx, 'wf_next')       // a is due
    await call(ctx, 'wf_done', { node: 'a' })
    await call(ctx, 'wf_next')       // gate is due
    await call(ctx, 'wf_verdict', { gate: 'gate', verdict: 'reloop' }) // iteration 1
    await call(ctx, 'wf_next')       // a is due again
    await call(ctx, 'wf_done', { node: 'a' })
    await call(ctx, 'wf_next')       // gate is due
    await call(ctx, 'wf_verdict', { gate: 'gate', verdict: 'reloop' }) // iteration 2
    await call(ctx, 'wf_next')       // a is due again
    await call(ctx, 'wf_done', { node: 'a' })
    await call(ctx, 'wf_next')       // gate is due
    await call(ctx, 'wf_verdict', { gate: 'gate', verdict: 'reloop' }) // iteration 3 > cap=2 → halts
    // The spine is now halted
    const next = await call(ctx, 'wf_next') as { status: string; halted?: { reason: string } }
    expect(next.status).toBe('done')
    expect(next.halted?.reason).toBe('max_iterations')
    // wf_redo should refuse
    await expect(call(ctx, 'wf_redo', { node: 'a' }))
      .rejects.toThrow(/iteration limit/)
    await expect(call(ctx, 'wf_redo', { node: 'a' }))
      .rejects.toThrow(/refactory graph-control rerun --acknowledge-failure/)
  })

  it('wf_redo refuses when run has failed', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir })
    // Complete a so it can be redone
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'a' })
    // Simulate a failed run by writing the status file
    mkdirSync(join(workdir, '.refactory'), { recursive: true })
    writeFileSync(join(workdir, '.refactory', 'spine-status.json'),
      JSON.stringify({ status: 'failed', error: 'authoring unresolved after 3 attempt(s)' }))
    // wf_redo should refuse
    await expect(call(ctx, 'wf_redo', { node: 'a' }))
      .rejects.toThrow(/CEO redo cannot/)
    await expect(call(ctx, 'wf_redo', { node: 'a' }))
      .rejects.toThrow(/refactory graph-control rerun --acknowledge-failure/)
  })

  it('exit code 3 with marker records failed and CEO tools refuse', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a',
             command: `bash -c 'echo "--- refactory failure ---" >&2; echo "BuildLoopStop: limit reached" >&2; exit 3'`,
             reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string }) => {
        // Shell out to run the actual command
        const result = spawnSync('bash', ['-c', spec.command], {
          encoding: 'utf8', timeout: 10_000,
        })
        return { result: async () => ({
          exitCode: result.status ?? 1, signal: result.signal, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: result.stdout ?? '', truncated: false },
          stderr: { text: result.stderr ?? '', truncated: false },
        }) }
      },
    })
    // wf_next drains and reports the failure
    const next = await call(ctx, 'wf_next') as { status: string; error?: string; node?: string }
    expect(next.status).toBe('failed')
    expect(next.node).toBe('a')
    // The recorded status must be 'failed' (not 'ready')
    const statusFile = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(statusFile.status).toBe('failed')
    expect(statusFile.last_failure).toBeDefined()
    expect(statusFile.last_failure.error).toMatch(/limit reached/)
    // wf_redo should refuse
    await expect(call(ctx, 'wf_redo', { node: 'a' }))
      .rejects.toThrow(/CEO redo cannot/)
    await expect(call(ctx, 'wf_redo', { node: 'a' }))
      .rejects.toThrow(/refactory graph-control rerun --acknowledge-failure/)
    // wf_edit skip should refuse
    await expect(call(ctx, 'wf_edit', { node: 'b', action: 'skip' }))
      .rejects.toThrow(/CEO redo cannot/)
  })

  it('exit code 3 without marker records ready (dm-95 unchanged)', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a',
             command: `bash -c 'echo "pytest internal error" >&2; exit 3'`,
             reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string }) => {
        const result = spawnSync('bash', ['-c', spec.command], {
          encoding: 'utf8', timeout: 10_000,
        })
        return { result: async () => ({
          exitCode: result.status ?? 1, signal: result.signal, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: result.stdout ?? '', truncated: false },
          stderr: { text: result.stderr ?? '', truncated: false },
        }) }
      },
    })
    const next = await call(ctx, 'wf_next') as { status: string }
    expect(next.status).toBe('failed')
    // The recorded status must be 'ready' — no marker means retryable
    const statusFile = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(statusFile.status).toBe('ready')
    expect(statusFile.last_failure).toBeDefined()
    // wf_redo should NOT throw (it is retryable)
    // Cannot redo 'a' since it never completed — just check status is ready
  })

  it('exit code 1 records ready (dm-95 regression guard)', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a',
             command: `bash -c 'echo "node error" >&2; exit 1'`,
             reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string }) => {
        const result = spawnSync('bash', ['-c', spec.command], {
          encoding: 'utf8', timeout: 10_000,
        })
        return { result: async () => ({
          exitCode: result.status ?? 1, signal: result.signal, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: result.stdout ?? '', truncated: false },
          stderr: { text: result.stderr ?? '', truncated: false },
        }) }
      },
    })
    const next = await call(ctx, 'wf_next') as { status: string }
    expect(next.status).toBe('failed')
    const statusFile = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(statusFile.status).toBe('ready')
    expect(statusFile.last_failure).toBeDefined()
  })

  it('failureDetailOf falls back to head+tail when no marker', async () => {
    const workdir = scratch()
    // A 3000-char stderr without marker — more than STATUS_DETAIL
    const head = 'E'.repeat(500)
    const middle = 'M'.repeat(2000)
    const tail = 'T'.repeat(500)
    const fullStderr = head + middle + tail
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a',
             command: 'bash -c "exit 1"',
             reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    // We test failureDetailOf's fallback through its effect on the recorded error
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (_spec: { command: string }) => {
        return { result: async () => ({
          exitCode: 1, signal: null, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: '', truncated: false },
          stderr: { text: fullStderr, truncated: false },
        }) }
      },
    })
    const next = await call(ctx, 'wf_next') as { status: string; error?: string }
    expect(next.status).toBe('failed')
    // The error should contain the head (first 500 chars)
    expect(next.error).toContain(head)
    // The error should contain at least part of the tail
    expect(next.error).toContain(tail)
    // The error should not be tail-only — it must start with the node id
    expect(next.error).toMatch(/^a exited/)
  })

  it('acknowledged rerun after exit-3 stop runs the stopped node again', async () => {
    const workdir = scratch()
    let callCount = 0
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a',
             command: `bash -c 'echo "--- refactory failure ---" >&2; echo "limit reached" >&2; exit 3'`,
             reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string }) => {
        callCount += 1
        const result = spawnSync('bash', ['-c', spec.command], {
          encoding: 'utf8', timeout: 10_000,
        })
        return { result: async () => ({
          exitCode: result.status ?? 1, signal: result.signal, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: result.stdout ?? '', truncated: false },
          stderr: { text: result.stderr ?? '', truncated: false },
        }) }
      },
    })
    // First wf_next: node 'a' fails with exit 3 + marker → status 'failed'
    await call(ctx, 'wf_next')
    const statusFile = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(statusFile.status).toBe('failed')
    // Queue a rerun control (simulates graph-control rerun --acknowledge-failure)
    const revision = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    writeRerunControl(workdir, 'a', { revision })
    // Next wf_next: applyControls processes the rerun, redo reopens 'a', drain re-runs it
    const initialCount = callCount
    const next2 = await call(ctx, 'wf_next') as { status: string }
    // 'a' ran again (drain executed it after redo)
    expect(callCount).toBeGreaterThan(initialCount)
    // It fails again with exit 3 (same command), but the point is it ran
    expect(next2.status).toBe('failed')
  })

  it('failureDetailOf with run warnings before marker shows reason first', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a',
             command: 'true',
             reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (_spec: { command: string }) => {
        // Simulate: run warning, then marker, then failure reason
        const stderr = 'WARNING: something harmless\n--- refactory failure ---\nauthoring unresolved after 3 attempt(s)\nFAILED tests/test_create.py::test_review'
        return { result: async () => ({
          exitCode: 3, signal: null, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: '', truncated: false },
          stderr: { text: stderr, truncated: false },
        }) }
      },
    })
    const next = await call(ctx, 'wf_next') as { status: string; error?: string }
    expect(next.status).toBe('failed')
    // The error should contain the reason first, not the warning
    expect(next.error).toContain('authoring unresolved')
    expect(next.error).toContain('FAILED tests/test_create.py::test_review')
    // The recorded status is 'failed' because exit 3 + marker
    const statusFile = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(statusFile.status).toBe('failed')
    // The last_failure error is the text after the marker: reason first, no warning
    expect(statusFile.last_failure.error).toMatch(/: authoring unresolved after 3 attempt\(s\)\nFAILED tests\/test_create\.py::test_review$/)
    expect(statusFile.last_failure.error).not.toContain('WARNING: something harmless')
    expect(next.error).not.toContain('WARNING: something harmless')
  })

  it('a person rerun of a completed upstream node after an exit-3 stop applies and the run continues', async () => {
    const workdir = scratch()
    const flag = join(workdir, 'FLAG')
    const graph: WorkflowGraph = {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'FnNode', id: 'a', command: `bash -c 'echo x >> ${workdir}/a-runs'`, reads: [], writes: [] },
        b: { _type: 'FnNode', id: 'b',
             command: `bash -c 'if [ -f ${flag} ]; then exit 0; fi; echo "--- refactory failure ---" >&2; echo "BuildLoopStop: limit" >&2; exit 3'`,
             reads: [], writes: [] },
        c: { _type: 'AgentNode', id: 'c', role: 'builder', prompt_template: 'C', reads: [], writes: [] },
      },
      edges: [{ source: 'a', target: 'b' }, { source: 'b', target: 'c' }],
    }
    const ctx = await mount(graph, { workdir, executeCommands: true })
    ctx.provide('shell', {
      resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string>; sandboxPolicy?: unknown }) => ({
        command: request.command, workdir: request.workdir ?? process.cwd(),
        timeoutMs: request.timeoutMs ?? 10_000, onExpiry: 'kill',
        stdoutMaxBytes: 1_000_000, env: request.env, sandboxPolicy: request.sandboxPolicy,
      }),
      execute: async (spec: { command: string }) => {
        const result = spawnSync('bash', ['-c', spec.command], { encoding: 'utf8', timeout: 10_000 })
        return { result: async () => ({
          exitCode: result.status ?? 1, signal: result.signal, timedOut: false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text: result.stdout ?? '', truncated: false },
          stderr: { text: result.stderr ?? '', truncated: false },
        }) }
      },
    })
    let next = await call(ctx, 'wf_next') as { status: string; node?: string }
    if (next.status !== 'failed') next = await call(ctx, 'wf_next') as { status: string; node?: string }
    expect(next.status).toBe('failed')
    expect(next.node).toBe('b')
    const status1 = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(status1.status).toBe('failed')
    await expect(call(ctx, 'wf_redo', { node: 'a' })).rejects.toThrow(/CEO redo cannot/)
    writeFileSync(flag, '')
    const revision = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    writeRerunControl(workdir, 'a', { revision })
    let after = await call(ctx, 'wf_next') as { status: string; nodes?: Array<{ id: string }> }
    for (let i = 0; i < 3 && after.status === 'ready' && !(after.nodes ?? []).some(n => n.id === 'c'); i++) {
      after = await call(ctx, 'wf_next') as { status: string; nodes?: Array<{ id: string }> }
    }
    const receipt = JSON.parse(readFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), 'utf8'))
    expect(receipt.status).toBe('applied')
    expect(readFileSync(join(workdir, 'a-runs'), 'utf8').trim().split('\n').length).toBe(2)
    expect(after.status).toBe('ready')
    expect((after.nodes ?? []).map(n => n.id)).toContain('c')
    const status2 = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(status2.status).not.toBe('failed')
  })

  it('a CEO redo asks the data bridge to keep the attempt facts', async () => {
    // The bridge is the Python reset that clears a reopened node's facts; the flag is
    // what keeps design mode's own attempt counts through the CEO's redo.
    const workdir = scratch()
    const payload = join(workdir, 'reset-payload.json')
    const ctx = await mount(agentGraph(), { workdir, dataCommand: `cat > '${payload}' && echo '{}' #` })
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'a' })
    await call(ctx, 'wf_redo', { node: 'a' })
    expect(JSON.parse(readFileSync(payload, 'utf8')).preserveAttemptFacts).toBe(true)
  })

  it('deferred rerun fails when invalidation plan changed after join', async () => {
    const workdir = scratch()
    const graph: WorkflowGraph = {
      name: 't', start_node: 'start',
      nodes: {
        start: { _type: 'AgentNode', id: 'start', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        fork: { _type: 'ForkNode', id: 'fork', targets: ['p', 'q'] },
        p: { _type: 'AgentNode', id: 'p', role: 'builder', prompt_template: 'P', reads: [], writes: [] },
        q: { _type: 'AgentNode', id: 'q', role: 'builder', prompt_template: 'Q', reads: [], writes: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['p', 'q'] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'start', target: 'fork' },
        { source: 'fork', target: 'join' },
        { source: 'join', target: 'end' },
      ],
    }
    const ctx = await mount(graph, { workdir })
    // Advance through start, fork, and complete one branch.
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'start' })
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'fork' })
    await call(ctx, 'wf_next')
    await call(ctx, 'wf_done', { node: 'p' })
    // wf_next writes checkpoint with p completed, q still pending.
    await call(ctx, 'wf_next')
    const revision = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-checkpoint.json'), 'utf8')).graph_revision
    // Queue a deferred rerun control with a stale deferred_nodes list that
    // does NOT match what planRedo('p') will return after the join — simulating
    // the scenario where the graph expanded between preview and apply.
    writeRerunControl(workdir, 'p', { revision, deferredNodes: ['p', 'join', 'end', 'phantom'] })
    // wf_next — q still due, concurrent work running, control deferred.
    await call(ctx, 'wf_next')
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(true)
    // Complete q so the join drains and concurrent work ends.
    await call(ctx, 'wf_done', { node: 'q' })
    // wf_next — hasConcurrentWork() is now false. applyControls compares the
    // deferred_nodes to the current planRedo('p').nodes, finds a mismatch, and
    // writes a failed receipt.
    await call(ctx, 'wf_next')
    // The control file should have been consumed (removed after failure).
    expect(existsSync(join(workdir, '.refactory', 'graph-controls', '1.json'))).toBe(false)
    // The receipt should report plan-changed failure.
    const receipt = JSON.parse(readFileSync(join(workdir, '.refactory', 'graph-control-receipt.json'), 'utf8'))
    expect(receipt.status).toBe('failed')
    expect(receipt.message).toContain('plan changed')
    // The node should NOT have been redone — p stays completed.
    const state = await call(ctx, 'wf_state') as { completed: string[] }
    expect(state.completed).toContain('p')
  })
})

function designGraph(): WorkflowGraph {
  return {
    name: 't', start_node: 'emit',
    nodes: {
      emit: { _type: 'AgentNode', id: 'emit', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      materialise: { _type: 'FnNode', id: 'materialise', command: 'true', reads: [], writes: [] },
      check: { _type: 'FnNode', id: 'check', command: 'true', reads: [], writes: [] },
      fork_qa: { _type: 'ForkNode', id: 'fork_qa', targets: ['health_checker', 'code_reviewer', 'qa_director'] },
      health_checker: { _type: 'AgentNode', id: 'health_checker', role: 'reviewer', prompt_template: 'HC', reads: [], writes: [] },
      code_reviewer: { _type: 'AgentNode', id: 'code_reviewer', role: 'reviewer', prompt_template: 'CR', reads: [], writes: [] },
      qa_director: { _type: 'AgentNode', id: 'qa_director', role: 'reviewer', prompt_template: 'QD', reads: [], writes: [] },
      join_qa: { _type: 'JoinNode', id: 'join_qa', sources: ['health_checker', 'code_reviewer', 'qa_director'] },
      gate_qa: { _type: 'GateNode', id: 'gate_qa', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
      report: { _type: 'AgentNode', id: 'report', role: 'builder', prompt_template: 'R', reads: [], writes: [] },
      deliver: { _type: 'AgentNode', id: 'deliver', role: 'builder', prompt_template: 'D', reads: [], writes: [] },
    },
    edges: [
      { source: 'emit', target: 'materialise' },
      { source: 'materialise', target: 'check' },
      { source: 'check', target: 'fork_qa' },
      { source: 'fork_qa', target: 'join_qa' },
      { source: 'join_qa', target: 'gate_qa' },
      { source: 'gate_qa', target: 'emit', condition: 'reloop' },
      { source: 'gate_qa', target: 'report', condition: 'proceed' },
      { source: 'report', target: 'deliver' },
    ],
  }
}

describe('cursor merge on redo', () => {
  it('redo of a branch keeps unaffected cursor entries and the run completes at deliver', () => {
    const spine = new Spine(designGraph())
    // First pass: emit → materialise → check → fork_qa → branches → join_qa → gate_qa reloop
    spine.complete('emit')
    spine.complete('materialise')
    spine.complete('check')
    spine.complete('fork_qa')
    spine.complete('health_checker')
    spine.complete('code_reviewer')
    spine.complete('qa_director')
    spine.complete('join_qa')
    spine.verdict('gate_qa', 'reloop')
    // Second pass through the loop body — complete everything up through the branches.
    // After the reloop, uncompleteBetween clears the loop body's completion records,
    // so code_reviewer must be re-completed in this iteration before it can be redone.
    spine.complete('emit')
    spine.complete('materialise')
    spine.complete('check')
    spine.complete('fork_qa')
    spine.complete('health_checker')
    spine.complete('code_reviewer')
    spine.complete('qa_director')
    // All branches done; join_qa pops from afters to cursor.
    expect(ids(spine)).toEqual(['join_qa'])

    // Redo code_reviewer — it completed in this iteration; join_qa is due in cursor
    spine.redo('code_reviewer')

    // With the fix: join_qa moved to afters (it is in the affected set); only code_reviewer is in cursor
    expect(ids(spine)).toEqual(['code_reviewer'])

    // Complete the redo: code_reviewer, then join_qa pops from afters, gate proceed, report, deliver
    spine.complete('code_reviewer')
    expect(ids(spine)).toEqual(['join_qa'])
    spine.complete('join_qa')
    spine.verdict('gate_qa', 'proceed')
    spine.complete('report')
    spine.complete('deliver')
    expect(spine.next().status).toBe('done')
  })
})

describe('fork-branch redo reopens join and downstream', () => {
  function designGraphWithSynth(): WorkflowGraph {
    return {
      name: 't', start_node: 'start',
      nodes: {
        start: { _type: 'AgentNode', id: 'start', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        fork_qa: { _type: 'ForkNode', id: 'fork_qa', targets: ['health_checker', 'code_reviewer', 'qa_director'] },
        health_checker: { _type: 'AgentNode', id: 'health_checker', role: 'reviewer', prompt_template: 'HC', reads: [], writes: [] },
        code_reviewer: { _type: 'AgentNode', id: 'code_reviewer', role: 'reviewer', prompt_template: 'CR', reads: [], writes: [] },
        qa_director: { _type: 'AgentNode', id: 'qa_director', role: 'reviewer', prompt_template: 'QD', reads: [], writes: [] },
        join_qa: { _type: 'JoinNode', id: 'join_qa', sources: ['health_checker', 'code_reviewer', 'qa_director'] },
        synth: { _type: 'AgentNode', id: 'synth', role: 'builder', prompt_template: 'SY', reads: [], writes: [] },
        gate_qa: { _type: 'GateNode', id: 'gate_qa', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'start', target: 'fork_qa' },
        { source: 'fork_qa', target: 'join_qa' },
        { source: 'join_qa', target: 'synth' },
        { source: 'synth', target: 'gate_qa' },
        { source: 'gate_qa', target: 'end', condition: 'proceed' },
      ],
    }
  }

  it('fork branch redo with design shape (with synth)', () => {
    const spine = new Spine(designGraphWithSynth())
    // Complete all nodes through end
    spine.complete('start')
    spine.complete('fork_qa')
    spine.complete('health_checker')
    spine.complete('code_reviewer')
    spine.complete('qa_director')
    spine.complete('join_qa')
    spine.complete('synth')
    spine.verdict('gate_qa', 'proceed')
    spine.complete('end')
    expect(spine.next().status).toBe('done')

    // planRedo should include join and downstream, not siblings
    const plan = spine.planRedo('code_reviewer')
    const planSet = new Set(plan.nodes)
    expect(planSet).toContain('code_reviewer')
    expect(planSet).toContain('join_qa')
    expect(planSet).toContain('synth')
    expect(planSet).toContain('gate_qa')
    expect(planSet).toContain('end')
    expect(planSet).not.toContain('health_checker')
    expect(planSet).not.toContain('qa_director')
    expect(planSet).not.toContain('fork_qa')
    expect(planSet).not.toContain('start')

    // redo
    spine.redo('code_reviewer')
    expect(ids(spine)).toEqual(['code_reviewer'])

    // Checkpoint round-trip: afters entry survives
    const cp = JSON.parse(JSON.stringify(spine.checkpoint()))
    const fresh = new Spine(designGraphWithSynth())
    fresh.restore(cp)
    expect(ids(fresh)).toEqual(['code_reviewer'])

    // Walk through to completion
    spine.complete('code_reviewer')
    expect(ids(spine)).toEqual(['join_qa'])
    spine.complete('join_qa')
    expect(ids(spine)).toEqual(['synth'])
    spine.complete('synth')
    expect(ids(spine)).toEqual(['gate_qa'])
    spine.verdict('gate_qa', 'proceed')
    expect(ids(spine)).toEqual(['end'])
    spine.complete('end')
    expect(spine.next().status).toBe('done')

    // Siblings stayed completed
    expect(spine.state().completed).toContain('health_checker')
    expect(spine.state().completed).toContain('qa_director')
  })

  it('multi-node branch redo (first node)', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['a', 'c'] },
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
        c: { _type: 'AgentNode', id: 'c', role: 'builder', prompt_template: 'C', reads: [], writes: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['b', 'c'] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'fork', target: 'join' },
        { source: 'a', target: 'b' },
        { source: 'join', target: 'end' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('fork')
    spine.complete('a')
    spine.complete('b')
    spine.complete('c')
    spine.complete('join')
    spine.complete('end')
    expect(spine.next().status).toBe('done')

    const plan = spine.planRedo('a')
    const planSet = new Set(plan.nodes)
    expect(planSet).toContain('a')
    expect(planSet).toContain('b')
    expect(planSet).toContain('join')
    expect(planSet).toContain('end')
    expect(planSet).not.toContain('c')
    expect(planSet).not.toContain('fork')

    spine.redo('a')
    expect(ids(spine)).toEqual(['a'])
    spine.complete('a')
    expect(ids(spine)).toEqual(['b'])
    spine.complete('b')
    expect(ids(spine)).toEqual(['join'])
    spine.complete('join')
    expect(ids(spine)).toEqual(['end'])
    spine.complete('end')
    expect(spine.next().status).toBe('done')
    expect(spine.state().completed).toContain('c')
  })

  it('multi-node branch redo (last node)', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'fork',
      nodes: {
        fork: { _type: 'ForkNode', id: 'fork', targets: ['a', 'c'] },
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
        c: { _type: 'AgentNode', id: 'c', role: 'builder', prompt_template: 'C', reads: [], writes: [] },
        join: { _type: 'JoinNode', id: 'join', sources: ['b', 'c'] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'fork', target: 'join' },
        { source: 'a', target: 'b' },
        { source: 'join', target: 'end' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('fork')
    spine.complete('a')
    spine.complete('b')
    spine.complete('c')
    spine.complete('join')
    spine.complete('end')
    expect(spine.next().status).toBe('done')

    const plan = spine.planRedo('b')
    const planSet = new Set(plan.nodes)
    expect(planSet).toContain('b')
    expect(planSet).toContain('join')
    expect(planSet).toContain('end')
    expect(planSet).not.toContain('a')
    expect(planSet).not.toContain('c')
    expect(planSet).not.toContain('fork')

    spine.redo('b')
    expect(ids(spine)).toEqual(['b'])
    spine.complete('b')
    expect(ids(spine)).toEqual(['join'])
    spine.complete('join')
    expect(ids(spine)).toEqual(['end'])
    spine.complete('end')
    expect(spine.next().status).toBe('done')
    expect(spine.state().completed).toContain('a')
    expect(spine.state().completed).toContain('c')
  })

  it('nested fork redo reopens both joins', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'outer_fork',
      nodes: {
        outer_fork: { _type: 'ForkNode', id: 'outer_fork', targets: ['inner_fork', 'z'] },
        inner_fork: { _type: 'ForkNode', id: 'inner_fork', targets: ['x', 'y'] },
        x: { _type: 'AgentNode', id: 'x', role: 'builder', prompt_template: 'X', reads: [], writes: [] },
        y: { _type: 'AgentNode', id: 'y', role: 'builder', prompt_template: 'Y', reads: [], writes: [] },
        inner_join: { _type: 'JoinNode', id: 'inner_join', sources: ['x', 'y'] },
        tail: { _type: 'AgentNode', id: 'tail', role: 'builder', prompt_template: 'T', reads: [], writes: [] },
        z: { _type: 'AgentNode', id: 'z', role: 'builder', prompt_template: 'Z', reads: [], writes: [] },
        outer_join: { _type: 'JoinNode', id: 'outer_join', sources: ['tail', 'z'] },
        end_node: { _type: 'AgentNode', id: 'end_node', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'outer_fork', target: 'outer_join' },
        { source: 'inner_fork', target: 'inner_join' },
        { source: 'inner_join', target: 'tail' },
        { source: 'outer_join', target: 'end_node' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('outer_fork')
    // inner_fork and z are parallel. Complete z first (though with dm-73 fixed,
    // the order no longer matters — both orderings work). This test is about redo.
    spine.complete('z')
    spine.complete('inner_fork')
    // x and y are parallel from inner fork
    spine.complete('x')
    spine.complete('y')
    // inner_join pops from afters
    spine.complete('inner_join')
    spine.complete('tail')
    // outer_join pops from afters
    spine.complete('outer_join')
    spine.complete('end_node')
    expect(spine.next().status).toBe('done')

    const plan = spine.planRedo('x')
    const planSet = new Set(plan.nodes)
    expect(planSet).toContain('x')
    expect(planSet).toContain('inner_join')
    expect(planSet).toContain('tail')
    expect(planSet).toContain('outer_join')
    expect(planSet).toContain('end_node')
    expect(planSet).not.toContain('y')
    expect(planSet).not.toContain('z')
    expect(planSet).not.toContain('inner_fork')
    expect(planSet).not.toContain('outer_fork')

    spine.redo('x')
    expect(ids(spine)).toEqual(['x'])
    spine.complete('x')
    expect(ids(spine)).toEqual(['inner_join'])
    spine.complete('inner_join')
    expect(ids(spine)).toEqual(['tail'])
    spine.complete('tail')
    expect(ids(spine)).toEqual(['outer_join'])
    spine.complete('outer_join')
    expect(ids(spine)).toEqual(['end_node'])
    spine.complete('end_node')
    expect(spine.next().status).toBe('done')
    expect(spine.state().completed).toContain('y')
    expect(spine.state().completed).toContain('z')
  })

  it('expanded director child redo reopens successor', () => {
    const graph: WorkflowGraph = {
      name: 't', start_node: 'director',
      nodes: {
        director: {
          _type: 'AgentNode', id: 'director', role: 'builder', prompt_template: 'D',
          reads: [], writes: [],
          expand: { plan: 'plan.json', role: 'worker', slug_field: 'slug', prompt_field: 'prompt', writes_template: '' },
        },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'director', target: 'end' },
      ],
    }
    const spine = new Spine(graph)
    expect(ids(spine)).toEqual(['director'])
    spine.expand('director', [{ slug: 'a', prompt: 'A' }, { slug: 'b', prompt: 'B' }])
    expect(ids(spine)).toEqual(expect.arrayContaining(['director__a', 'director__b']))
    spine.complete('director__a')
    spine.complete('director__b')
    // end pops from afters
    expect(ids(spine)).toEqual(['end'])
    spine.complete('end')
    expect(spine.next().status).toBe('done')

    const plan = spine.planRedo('director__a')
    const planSet = new Set(plan.nodes)
    expect(planSet).toContain('director__a')
    expect(planSet).toContain('end')
    expect(planSet).not.toContain('director__b')
    expect(planSet).not.toContain('director')

    spine.redo('director__a')
    expect(ids(spine)).toEqual(['director__a'])
    spine.complete('director__a')
    expect(ids(spine)).toEqual(['end'])
    spine.complete('end')
    expect(spine.next().status).toBe('done')
    expect(spine.state().completed).toContain('director__b')
  })

  it('continuation does not depend on join sources', () => {
    // join_qa's sources names only health_checker, but code_reviewer is also a
    // fork branch. The continuation comes from the fork's unconditional target,
    // not from the join's sources list.
    const graph: WorkflowGraph = {
      name: 't', start_node: 'start',
      nodes: {
        start: { _type: 'AgentNode', id: 'start', role: 'builder', prompt_template: 'S', reads: [], writes: [] },
        fork_qa: { _type: 'ForkNode', id: 'fork_qa', targets: ['health_checker', 'code_reviewer', 'qa_director'] },
        health_checker: { _type: 'AgentNode', id: 'health_checker', role: 'reviewer', prompt_template: 'HC', reads: [], writes: [] },
        code_reviewer: { _type: 'AgentNode', id: 'code_reviewer', role: 'reviewer', prompt_template: 'CR', reads: [], writes: [] },
        qa_director: { _type: 'AgentNode', id: 'qa_director', role: 'reviewer', prompt_template: 'QD', reads: [], writes: [] },
        join_qa: { _type: 'JoinNode', id: 'join_qa', sources: ['health_checker'] },
        synth: { _type: 'AgentNode', id: 'synth', role: 'builder', prompt_template: 'SY', reads: [], writes: [] },
        gate_qa: { _type: 'GateNode', id: 'gate_qa', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        end: { _type: 'AgentNode', id: 'end', role: 'builder', prompt_template: 'E', reads: [], writes: [] },
      },
      edges: [
        { source: 'start', target: 'fork_qa' },
        { source: 'fork_qa', target: 'join_qa' },
        { source: 'join_qa', target: 'synth' },
        { source: 'synth', target: 'gate_qa' },
        { source: 'gate_qa', target: 'end', condition: 'proceed' },
      ],
    }
    const spine = new Spine(graph)
    spine.complete('start')
    spine.complete('fork_qa')
    spine.complete('health_checker')
    spine.complete('code_reviewer')
    spine.complete('qa_director')
    spine.complete('join_qa')
    spine.complete('synth')
    spine.verdict('gate_qa', 'proceed')
    spine.complete('end')
    expect(spine.next().status).toBe('done')

    // Even though join_qa.sources only names health_checker, redo of code_reviewer
    // still reopens join_qa and downstream because the continuation comes from
    // the fork's unconditional target, not from join sources.
    const plan = spine.planRedo('code_reviewer')
    const planSet = new Set(plan.nodes)
    expect(planSet).toContain('code_reviewer')
    expect(planSet).toContain('join_qa')
    expect(planSet).toContain('synth')
    expect(planSet).toContain('gate_qa')
    expect(planSet).toContain('end')

    spine.redo('code_reviewer')
    expect(ids(spine)).toEqual(['code_reviewer'])
    spine.complete('code_reviewer')
    expect(ids(spine)).toEqual(['join_qa'])
    spine.complete('join_qa')
    expect(ids(spine)).toEqual(['synth'])
    spine.complete('synth')
    expect(ids(spine)).toEqual(['gate_qa'])
    spine.verdict('gate_qa', 'proceed')
    expect(ids(spine)).toEqual(['end'])
    spine.complete('end')
    expect(spine.next().status).toBe('done')
  })
})

describe('iteration preservation on CEO redo', () => {
  function gateLoopGraph(): WorkflowGraph {
    return {
      name: 't', start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', reads: [], writes: [] },
        gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'fn', evaluator_command: 'true', reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', reads: [], writes: [] },
      },
      edges: [
        { source: 'a', target: 'gate' },
        { source: 'gate', target: 'a', condition: 'reloop' },
        { source: 'gate', target: 'b', condition: 'proceed' },
      ],
    }
  }

  it('CEO redo preserves gate iteration count', () => {
    const spine = new Spine(gateLoopGraph(), { maxIterations: 2 })
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 1
    expect((spine.state().iterations as Record<string, number>).gate).toBe(1)
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 2
    expect((spine.state().iterations as Record<string, number>).gate).toBe(2)
    // After the reloop, a is uncompleted by uncompleteBetween. Complete it again
    // so planRedo accepts it.
    spine.complete('a')
    // CEO redo with preserveIterations
    spine.redo('a', { preserveIterations: true })
    expect((spine.state().iterations as Record<string, number>).gate).toBe(2)
    // One more reloop → count = 3 > cap = 2 → halts
    spine.complete('a')
    spine.verdict('gate', 'reloop')
    expect(spine.next().status).toBe('done')
    expect((spine.next() as { halted?: { reason: string } }).halted?.reason).toBe('max_iterations')
  })

  it('graph-control rerun resets gate iteration count', () => {
    const spine = new Spine(gateLoopGraph(), { maxIterations: 2 })
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 1
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 2
    expect((spine.state().iterations as Record<string, number>).gate).toBe(2)
    // After the reloop, a is uncompleted. Complete it again so planRedo accepts it.
    spine.complete('a')
    // Rerun redo without preserveIterations (the default, as graph-control uses)
    spine.redo('a')
    expect((spine.state().iterations as Record<string, number>).gate).toBeUndefined()
    // Now 3 reloops needed to hit the cap (count > 2)
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 1
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 2
    spine.complete('a')
    spine.verdict('gate', 'reloop') // iteration 3 > 2 → halts
    expect(spine.next().status).toBe('done')
    expect((spine.next() as { halted?: { reason: string } }).halted?.reason).toBe('max_iterations')
  })
})
