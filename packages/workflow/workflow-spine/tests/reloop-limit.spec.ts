/**
 * A person's reloops at a user gate keep their limit (dm-141), but the card says how
 * many are left, and the halt says the limit was reached and how to go on.
 */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import UserQuestionService from '@deepseek-ai/dsh-user-questions'
import * as workflowSpine from '../src/index.ts'
import { presentResult, presentationMeta } from '../src/present.ts'
import type { WorkflowGraph } from '../src/types.ts'

const graph: WorkflowGraph = {
  name: 'reloop-limit',
  start_node: 'design_doc',
  nodes: {
    design_doc: { _type: 'FnNode', id: 'design_doc', content: 'design', writes: [] },
    gate_strategy: {
      _type: 'GateNode', id: 'gate_strategy', evaluator_type: 'user', reads: [], max_iterations: 2,
      gate_prompt: 'Approve the design?',
    },
    emit: { _type: 'FnNode', id: 'emit', writes: [], command: '' },
  },
  edges: [
    { source: 'design_doc', target: 'gate_strategy' },
    { source: 'gate_strategy', target: 'emit', condition: 'proceed' },
    { source: 'gate_strategy', target: 'design_doc', condition: 'reloop' },
  ],
}

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

async function run(ctx: Context, name: string, args: unknown): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} not registered`)
  return definition.execute(args, {} as never)
}

/** A run whose person gives `answer` at the design gate every time it is asked. */
async function answering(workdir: string, asked: string[], answer = 'reloop', custom = ''): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(UserQuestionService)
  ctx.on('user-questions/request', request => Promise.resolve({
    answers: request.questions.map(question => {
      asked.push(question.question)
      return { id: question.id, selected: [answer], custom }
    }),
  }))
  await ctx.plugin(workflowSpine, { graph, toolPrefix: 'wf', maxIterations: 9, workdir })
  return ctx
}

/** Run the design step, then put the gate to the person; return what the gate call returned. */
async function askOnce(ctx: Context): Promise<Record<string, unknown>> {
  await run(ctx, 'wf_next', {})
  await run(ctx, 'wf_done', { node: 'design_doc' })
  await run(ctx, 'wf_next', {})
  return await run(ctx, 'wf_verdict', { gate: 'gate_strategy' }) as Record<string, unknown>
}

describe('the reloop limit at a user gate', () => {
  it('shows the reloops left on the card, then says the limit stopped the run and how to go on', async () => {
    const workdir = mkdtempSync(join(tmpdir(), 'dsh-spine-reloop-limit-'))
    dirs.push(workdir)
    const asked: string[] = []
    const ctx = await answering(workdir, asked)

    expect(await askOnce(ctx)).not.toHaveProperty('halted')
    expect(await askOnce(ctx)).not.toHaveProperty('halted')
    const last = await askOnce(ctx)

    expect(asked).toHaveLength(3)
    expect(asked[0]).toBe('Approve the design? Reloops left: 2 of 2.')
    expect(asked[1]).toBe('Approve the design? Reloops left: 1 of 2.')
    expect(asked[2]).toBe('Approve the design? Reloops left: 0 of 2. '
      + 'A reloop now stops the run instead of going back to design_doc.')

    // The limit holds: the third reloop stops the run, and says why and what to do.
    expect(last).toMatchObject({ status: 'routed', verdict: 'reloop', answered_by: 'user', halted: true })
    const message = String(last.message)
    expect(message).toContain('The run stopped: gate_strategy reached its limit of 2 reloops.')
    expect(message).toContain(`refactory graph-control preview --run ${workdir} --node design_doc`)
    expect(message).toContain(`refactory graph-control rerun --run ${workdir} --node design_doc --revision REVISION`)
    expect(message).toContain('A rerun by a person resets the limit.')

    // Every later status write keeps the message, where `refactory status` and the
    // interactive CLI read it.
    await run(ctx, 'wf_next', {})
    const recorded = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(recorded).toMatchObject({ status: 'halted', error: message, wait_reason: message })

    const view = presentResult('verdict', { gate: 'gate_strategy' },
      { isError: false, meta: presentationMeta('verdict', last) })
    expect(view).toEqual({ card: 'generic', title: 'Reloop limit reached: the run stopped at gate_strategy' })
  })

  it('says nothing about a limit when the person stops the run', async () => {
    const workdir = mkdtempSync(join(tmpdir(), 'dsh-spine-reloop-limit-'))
    dirs.push(workdir)
    const ctx = await answering(workdir, [], 'halt')
    const stopped = await askOnce(ctx)
    expect(stopped).toMatchObject({ status: 'routed', verdict: 'halt', answered_by: 'user', halted: true })
    expect(stopped).not.toHaveProperty('message')
    const recorded = JSON.parse(readFileSync(join(workdir, '.refactory', 'spine-status.json'), 'utf8'))
    expect(recorded.status).toBe('halted')
    expect(recorded).not.toHaveProperty('error')
    const view = presentResult('verdict', { gate: 'gate_strategy' },
      { isError: false, meta: presentationMeta('verdict', stopped) })
    expect(view).toEqual({ card: 'generic', title: 'You stopped the run' })
  })

  it('says nothing about a limit when the person stops the run with a reason', async () => {
    const workdir = mkdtempSync(join(tmpdir(), 'dsh-spine-reloop-limit-'))
    dirs.push(workdir)
    const ctx = await answering(workdir, [], 'halt', 'not worth building')
    const stopped = await askOnce(ctx)
    // Their reason is the reply's feedback; no halt message repeats it as the gate's.
    expect(stopped).toMatchObject({ status: 'routed', verdict: 'halt', answered_by: 'user', halted: true,
      feedback: 'not worth building' })
    expect(stopped).not.toHaveProperty('message')
  })
})
