import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import UserQuestionService from '@deepseek-ai/dsh-user-questions'
import * as workflowSpine from '../src/index.ts'
import type { WorkflowGraph } from '../src/types.ts'

/**
 * Minimal graph: FnNode \u2192 USER gate (proceed/reloop) \u2192 done.
 * The gate has two labelled edges so `acceptedVerdicts` returns ['proceed', 'reloop'].
 */
const graph: WorkflowGraph = {
  name: 'freetext-gate-test',
  start_node: 'init',
  nodes: {
    init: { _type: 'FnNode', id: 'init', content: 'setup', writes: ['intent.md'] },
    gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', reads: [] },
    done: { _type: 'FnNode', id: 'done', writes: [], command: '' },
  },
  edges: [
    { source: 'init', target: 'gate' },
    { source: 'gate', target: 'done', condition: 'proceed' },
    { source: 'gate', target: 'init', condition: 'reloop' },
  ],
}

async function run(ctx: Context, name: string, args: unknown): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} not registered`)
  return definition.execute(args, {} as never)
}

function ids(result: unknown): string[] {
  const nodes = (result as { nodes?: Array<{ id: string }> }).nodes ?? []
  return nodes.map(node => node.id)
}

describe('user gate free-text answers', () => {
  function setup(personAnswers: Array<{ selected: string[]; custom: string }>) {
    const ctx = new Context()
    const pluginReady = (async () => {
      await ctx.plugin(SystemPrompt)
      await ctx.plugin(ToolRuntime)
      await ctx.plugin(UserQuestionService)
      ctx.on('user-questions/request', request => Promise.resolve({
        answers: request.questions.map(question => {
          const answer = personAnswers.shift()!
          return { id: question.id, selected: answer.selected, custom: answer.custom }
        }),
      }))
      await ctx.plugin(workflowSpine, { graph, toolPrefix: 'wf', maxIterations: 3 })
    })()
    return { ctx, pluginReady }
  }

  async function advanceToGate(ctx: Context): Promise<void> {
    const next = await run(ctx, 'wf_next', {})
    if (ids(next).includes('init')) {
      await run(ctx, 'wf_done', { node: 'init' })
      await run(ctx, 'wf_next', {})
    }
  }

  it('routes case-insensitive exact label match', async () => {
    const { ctx, pluginReady } = setup([
      { selected: [], custom: 'Proceed' },
    ])
    await pluginReady
    await advanceToGate(ctx)
    expect(ids(await run(ctx, 'wf_next', {}))).toEqual(['gate'])
    const result = await run(ctx, 'wf_verdict', { gate: 'gate' })
    expect(result).toMatchObject({ status: 'routed', verdict: 'proceed', answered_by: 'user' })
  })

  it('parses "reloop: feedback" prefix and routes with feedback', async () => {
    const { ctx, pluginReady } = setup([
      { selected: [], custom: 'reloop: make it simpler' },
    ])
    await pluginReady
    await advanceToGate(ctx)
    expect(ids(await run(ctx, 'wf_next', {}))).toEqual(['gate'])
    const result = await run(ctx, 'wf_verdict', { gate: 'gate' })
    expect(result).toMatchObject({
      status: 'routed',
      verdict: 'reloop',
      answered_by: 'user',
      feedback: 'make it simpler',
    })
  })

  it('returns replied for unrecognized free text, gate stays due', async () => {
    const { ctx, pluginReady } = setup([
      { selected: [], custom: 'show me the strategy' },
    ])
    await pluginReady
    await advanceToGate(ctx)
    expect(ids(await run(ctx, 'wf_next', {}))).toEqual(['gate'])
    const result = await run(ctx, 'wf_verdict', { gate: 'gate' })
    expect(result).toMatchObject({ status: 'replied', gate: 'gate', text: 'show me the strategy' })
    const state = await run(ctx, 'wf_state', {}) as { cursor: string[] }
    expect(state.cursor).toContain('gate')
  })
})
