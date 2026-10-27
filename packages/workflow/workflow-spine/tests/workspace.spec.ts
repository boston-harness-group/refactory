/**
 * Agent workspaces on the spine. The workspace itself is Python's: the dispatched
 * worker (`refactory_dsh/worker_node.py`) prepares it, runs the agent there and collects
 * the node's output before delivery, and Python's lowering refuses a workspace on the
 * CEO-subagent route (`workers.lower_workflow`). The spine only passes the declaration
 * through, running the node's dispatch command as it runs any other.
 */
import { describe, expect, it } from 'vitest'
import { Spine, validateGraph } from '../src/spine.ts'
import { commandOf } from '../src/execute.ts'
import type { WorkflowGraph, WorkflowNode } from '../src/types.ts'

const DISPATCH = 'python -m refactory_dsh.worker_node /run/.refactory/worker-specs/abc.json'

function graph(emit: Partial<WorkflowNode>): WorkflowGraph {
  return {
    name: 'design',
    start_node: 'emit',
    nodes: {
      emit: { _type: 'AgentNode', id: 'emit', role: 'builder', prompt_template: 'Make the change.', reads: [], writes: ['emit.txt'], ...emit },
      materialise: { _type: 'FnNode', id: 'materialise', command: 'true', reads: ['emit.txt'], writes: [] },
    },
    edges: [{ source: 'emit', target: 'materialise' }],
  }
}

describe('agent workspaces', () => {
  it('passes a dispatched node\'s workspace through and runs its dispatch command', () => {
    const declared = graph({
      workspace: { kind: 'clone', of: 'attempt', collect: 'diff' }, dispatch_command: DISPATCH, command_timeout_s: 9000,
    })
    expect(validateGraph(declared)).toEqual([])
    const spine = new Spine(declared)
    const next = spine.next()
    const view = next.status === 'ready' ? next.nodes[0] : undefined
    expect(view?.id).toBe('emit')
    // The plugin runs the worker; the worker prepares the workspace.
    expect(commandOf(view!)).toBe(DISPATCH)
    expect(view!.command_timeout_s).toBe(9000)
  })

  it('loads a graph whose agent nodes declare workspaces as it loads any other', () => {
    expect(validateGraph(graph({ workspace: { kind: 'scratch', collect: 'files' } }))).toEqual([])
    expect(validateGraph(graph({}))).toEqual([])
  })
})
