import { describe, it, expect } from 'vitest'
import { Spine } from '../src/spine'
import type { WorkflowGraph, WorkflowNode } from '../src/types'

function makeGraph(systemPrompt?: string): WorkflowGraph {
  const node: WorkflowNode = {
    _type: 'AgentNode',
    id: 'build',
    role: 'builder',
    prompt_template: 'do it',
    writes: ['out.txt'],
    reads: [],
  }
  if (systemPrompt !== undefined) {
    node.system_prompt = systemPrompt
  }
  return {
    name: 'test',
    start_node: 'build',
    nodes: { build: node },
    edges: [],
  }
}

describe('system_prompt on AgentNode', () => {
  it('describe() carries system_prompt in NodeView when the node has one', () => {
    const spine = new Spine(makeGraph('be careful'))
    const view = spine.view('build')
    expect(view.system_prompt).toBe('be careful')
  })

  it('describe() omits system_prompt from NodeView when the node has none', () => {
    const spine = new Spine(makeGraph())
    const view = spine.view('build')
    expect(view.system_prompt).toBeUndefined()
  })

  it('describe() omits system_prompt from NodeView when empty string', () => {
    const spine = new Spine(makeGraph(''))
    const view = spine.view('build')
    expect(view.system_prompt).toBeUndefined()
  })
})
