import { describe, expect, it } from 'vitest'
import { Spine } from '../src/spine.ts'
import type { WorkflowGraph } from '../src/types.ts'

describe('tool_policy on WorkflowNode', () => {
  const baseGraph: WorkflowGraph = {
    name: 'tp-test',
    start_node: 'a',
    nodes: {
      a: {
        _type: 'AgentNode',
        id: 'a',
        role: 'builder',
        prompt_template: 'do the thing',
        writes: ['out.txt'],
        reads: [],
        tool_policy: 'none',
      },
    },
    edges: [],
  }

  it('accepts tool_policy on WorkflowNode', () => {
    const spine = new Spine(baseGraph)
    const next = spine.next()
    expect(next.status).toBe('ready')
    expect(next.nodes?.[0]?.id).toBe('a')
  })

  it('missing tool_policy is default (tools enabled)', () => {
    const graph: WorkflowGraph = {
      ...baseGraph,
      nodes: {
        a: {
          _type: 'AgentNode',
          id: 'a',
          role: 'builder',
          prompt_template: 'do the thing',
          writes: ['out.txt'],
          reads: [],
          // no tool_policy
        },
      },
    }
    const spine = new Spine(graph)
    const next = spine.next()
    expect(next.nodes?.[0]?.tool_policy).toBeUndefined()
  })

  it('passes tool_policy through describe', () => {
    const spine = new Spine(baseGraph)
    const next = spine.next()
    // The view should include tool_policy: 'none' for the no-tools agent node
    expect(next.nodes?.[0]?.tool_policy).toBe('none')
  })

  it('invalid tool_policy is refused before dispatch', () => {
    // TypeScript types restrict tool_policy to 'none' | undefined, but at runtime
    // Runtime input must also refuse unsupported policies.
    const graph: WorkflowGraph = {
      ...baseGraph,
      nodes: {
        a: {
          _type: 'AgentNode',
          id: 'a',
          role: 'builder',
          prompt_template: 'do the thing',
          writes: ['out.txt'],
          reads: [],
          tool_policy: 'invalid' as any,
        },
      },
    }
    const spine = new Spine(graph)
    expect(() => spine.next()).toThrow('invalid tool_policy')
  })

  it('dispatched spine passes tool_policy to worker', () => {
    // A dispatched AgentNode has a dispatch_command; verify tool_policy still appears
    const graph: WorkflowGraph = {
      ...baseGraph,
      nodes: {
        a: {
          _type: 'AgentNode',
          id: 'a',
          role: 'builder',
          prompt_template: 'do the thing',
          writes: ['out.txt'],
          reads: [],
          tool_policy: 'none',
          dispatch_command: 'python -m refactory_dsh.worker_node spec.json',
          command_timeout_s: 300,
        },
      },
    }
    const spine = new Spine(graph)
    const next = spine.next()
    expect(next.nodes?.[0]?.tool_policy).toBe('none')
    expect(next.nodes?.[0]?.command).toBe('python -m refactory_dsh.worker_node spec.json')
  })
})
