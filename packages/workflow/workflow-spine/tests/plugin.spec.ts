import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import UserQuestionService from '@deepseek-ai/dsh-user-questions'
import * as workflowSpine from '../src/index.ts'
import type { WorkflowGraph } from '../src/types.ts'

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

const graphWithReads: WorkflowGraph = {
  name: 't',
  start_node: 'a',
  nodes: {
    a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
    gate: { _type: 'GateNode', id: 'gate', evaluator_type: 'user', gate_prompt: 'ok?', reads: ['design.md'] },
    b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
  },
  edges: [
    { source: 'a', target: 'gate' },
    { source: 'gate', target: 'b', condition: 'proceed' },
    { source: 'gate', target: 'a', condition: 'reloop' },
  ],
}

async function setup(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(UserQuestionService)
  ctx.on('user-questions/request', request => Promise.resolve({
    answers: request.questions.map(question => ({ id: question.id, selected: ['proceed'] })),
  }))
  await ctx.plugin(workflowSpine, { graph, toolPrefix: 'wf', maxIterations: 3 })
  return ctx
}

async function run(ctx: Context, name: string, args: unknown): Promise<unknown> {
  const definition = ctx.tools.get(name)
  if (definition === undefined) throw new Error(`tool ${name} not registered`)
  return definition.execute(args, {} as never)
}

/** One question as the user-questions channel received it. */
interface AskedQuestion {
  id: string
  question?: string
  detail?: string
  options?: Array<{ label: string }>
  intent?: unknown
}

/**
 * A composition walked to its user gate, whose answerer records each question and
 * replies with `reply`.
 * @param gateGraph - the graph to open.
 * @param workdir - the run directory, when the run has one.
 * @param reply - the person's answer to a question.
 * @returns the composition and the questions asked so far.
 */
async function gateRun(
  gateGraph: WorkflowGraph, workdir: string | undefined, reply: (question: AskedQuestion) => Promise<unknown>,
): Promise<{ ctx: Context; asked: AskedQuestion[] }> {
  const asked: AskedQuestion[] = []
  const ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(UserQuestionService)
  ctx.on('user-questions/request', request => {
    asked.push(...request.questions)
    return reply(request.questions[0]!) as never
  })
  await ctx.plugin(workflowSpine, { graph: gateGraph, toolPrefix: 'wf', maxIterations: 3, ...(workdir === undefined ? {} : { workdir }) })
  await run(ctx, 'wf_done', { node: 'a' })
  return { ctx, asked }
}

/** An answerer that always chooses `proceed`. */
function proceed(question: AskedQuestion): Promise<unknown> {
  return Promise.resolve({ answers: [{ id: question.id, selected: ['proceed'] }] })
}

describe('workflow-spine real composition', () => {
  it('registers the six wf_* tools and walks a graph through them', async () => {
    const ctx = await setup()
    for (const name of ['wf_next', 'wf_done', 'wf_verdict', 'wf_edit', 'wf_expand', 'wf_state']) {
      expect(ctx.tools.get(name)).toBeDefined()
    }

    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'a' }] })
    await run(ctx, 'wf_done', { node: 'a' })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'gate' }] })
    await run(ctx, 'wf_verdict', { gate: 'gate' })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'b' }] })
    await run(ctx, 'wf_done', { node: 'b' })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'done' })
  })

  it('refuses a verdict for a user gate instead of recording one the person never gave', async () => {
    const ctx = await setup()
    await run(ctx, 'wf_done', { node: 'a' })
    await expect(run(ctx, 'wf_verdict', { gate: 'gate', verdict: 'proceed' }))
      .rejects.toThrow(/user gate.*never gave/s)
    // The honest call — the gate alone — is the one that routes, and it routes the
    // person's answer, not the model's.
    expect(await run(ctx, 'wf_verdict', { gate: 'gate' }))
      .toMatchObject({ status: 'routed', gate: 'gate', verdict: 'proceed', answered_by: 'user' })
  })

  it('refuses to route a user gate when the composition can ask nobody', async () => {
    // The defect this holds: a headless run used to route a `user` gate with whatever
    // verdict the model passed, which is indistinguishable from one a person gave.
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(workflowSpine, { graph, toolPrefix: 'wf', maxIterations: 3 })
    await run(ctx, 'wf_done', { node: 'a' })
    await expect(run(ctx, 'wf_verdict', { gate: 'gate' }))
      .rejects.toThrow(/gate gate is answered by a person/)
  })

  it('refuses to route a user gate when no answerer accepts the request', async () => {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(UserQuestionService)
    await ctx.plugin(workflowSpine, { graph, toolPrefix: 'wf', maxIterations: 3 })
    await run(ctx, 'wf_done', { node: 'a' })
    await expect(run(ctx, 'wf_verdict', { gate: 'gate' }))
      .rejects.toThrow(/gate gate is answered by a person and could not be asked/)
  })

  it('asks with the gate prompt and outcomes only, plus halt', async () => {
    // The question stays host-neutral and short: no presentation intent, no detail and
    // no description, so the gate's own outcomes (and free text) are all there is.
    const tmp = mkdtempSync(join(tmpdir(), 'spine-gate-reads-'))
    try {
      const { ctx, asked } = await gateRun(graphWithReads, tmp, proceed)
      expect(await run(ctx, 'wf_verdict', { gate: 'gate' }))
        .toMatchObject({ status: 'routed', verdict: 'proceed', answered_by: 'user' })
      expect(asked[0]!.detail).toBeUndefined()
      expect(asked[0]!.intent).toBeUndefined()
      expect(asked[0]!.options).toEqual([{ label: 'proceed' }, { label: 'reloop' }, { label: 'halt' }])
    } finally { rmSync(tmp, { recursive: true, force: true }) }
  })

  it('fills the run\'s facts into the question it puts and the view the CEO gets', async () => {
    // Design mode's approval card names the repository delivery would create, and the
    // card shows only the gate's own question, so the fact is filled there.
    const withFact: WorkflowGraph = {
      ...graph,
      nodes: { ...graph.nodes, gate: { ...graph.nodes.gate!, gate_prompt: 'Approve? Then, {{delivery}}; {{unknown}} stays.' } },
    }
    const tmp = mkdtempSync(join(tmpdir(), 'spine-gate-facts-'))
    try {
      writeFileSync(join(tmp, 'facts.json'), JSON.stringify({
        delivery: 'once the reviews pass, this run creates the private GitHub repository octo/tool',
      }))
      const { ctx, asked } = await gateRun(withFact, tmp, proceed)
      const view = await run(ctx, 'wf_next', {}) as { nodes: Array<{ id: string; gate_prompt?: string }> }
      const filled = 'Approve? Then, once the reviews pass, this run creates the private GitHub repository '
        + 'octo/tool; {{unknown}} stays.'
      expect(view.nodes.find(node => node.id === 'gate')?.gate_prompt).toBe(filled)
      await run(ctx, 'wf_verdict', { gate: 'gate' })
      // The card also says how many reloops are left before one stops the run (dm-141).
      expect(asked[0]!.question).toBe(`${filled} Reloops left: 3 of 3.`)
      expect(asked[0]!.detail).toBeUndefined()
    } finally { rmSync(tmp, { recursive: true, force: true }) }
    expect(workflowSpine.withFacts('Then, {{delivery}}.', undefined)).toBe('Then, {{delivery}}.')
  })

  it('puts the gate\'s card in the detail and keeps the question to its one-line ask', async () => {
    // The question card shows `question` as a title that keeps no line breaks, and
    // renders `detail` as text: design mode's card goes there, line by line (dm-163).
    const card = 'Design: The tool prints the name.\n\nDesign review: PASS, nothing blocks the build\n\n'
      + 'Pitfalls: 2\n- PIT-001 (high): addressed\n- PIT-002 (low): not mentioned'
    const withCard: WorkflowGraph = {
      ...graph,
      nodes: { ...graph.nodes, gate: { ...graph.nodes.gate!, gate_prompt: '{{design_card}}\n\nApprove the design? Then, {{delivery}}.' } },
    }
    const tmp = mkdtempSync(join(tmpdir(), 'spine-gate-card-'))
    try {
      writeFileSync(join(tmp, 'facts.json'), JSON.stringify({ design_card: card, delivery: 'this run opens a pull request' }))
      const { ctx, asked } = await gateRun(withCard, tmp, proceed)
      await run(ctx, 'wf_next', {})
      await run(ctx, 'wf_verdict', { gate: 'gate' })
      expect(asked[0]!.question).toBe('Approve the design? Then, this run opens a pull request. Reloops left: 3 of 3.')
      expect(asked[0]!.detail).toBe(card)
    } finally { rmSync(tmp, { recursive: true, force: true }) }
  })

  it('lets the person stop the run at a user gate that has no halt edge', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'spine-gate-halt-'))
    try {
      const { ctx } = await gateRun(graphWithReads, tmp,
        question => Promise.resolve({ answers: [{ id: question.id, selected: [], custom: 'halt' }] }))
      expect(await run(ctx, 'wf_verdict', { gate: 'gate' }))
        .toMatchObject({ status: 'routed', verdict: 'halt', answered_by: 'user', halted: true })
      expect(await run(ctx, 'wf_next', {})).toMatchObject({ halted: { gate: 'gate', reason: 'verdict' } })
      expect(await run(ctx, 'wf_state', {})).toMatchObject({ gate_verdicts: { gate: 'halt' }, cursor: [] })
    } finally { rmSync(tmp, { recursive: true, force: true }) }
  })

  it('tells the CEO to reply, then ask the gate again at once, when the answer is no outcome', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'spine-gate-replied-'))
    try {
      const { ctx } = await gateRun(graphWithReads, tmp,
        question => Promise.resolve({ answers: [{ id: question.id, selected: [], custom: 'what happens next?' }] }))
      const replied = await run(ctx, 'wf_verdict', { gate: 'gate' }) as { status: string; note: string }
      expect(replied.status).toBe('replied')
      expect(replied.note).toMatch(/asked correctly/)
      expect(replied.note).toMatch(/call this tool again with the gate alone straight away/)
      expect(replied.note).toMatch(/Do not ask them to type an outcome in chat/)
    } finally { rmSync(tmp, { recursive: true, force: true }) }
  })

  it('CEO prompt instructs to present gate reads files before routing a user gate', async () => {
    const ctx = await setup()
    const assembly = await ctx.systemPrompt.assemble()
    const spineSection = assembly.sections.find(s => s.name === 'tool:wf-spine')
    expect(spineSection).toBeDefined()
    expect(spineSection!.text).toMatch(/user gate/)
    expect(spineSection!.text).toMatch(/reads/)
    expect(spineSection!.text).toMatch(/summarise|summarize/)
  })

  it('CEO system prompt includes delegation and review discipline rules (Sacred Rules 8/9)', async () => {
    // Use a non-default tool prefix so a hardcoded 'wf_' fails the test.
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(UserQuestionService)
    ctx.on('user-questions/request', request => Promise.resolve({
      answers: request.questions.map(question => ({ id: question.id, selected: ['proceed'] })),
    }))
    await ctx.plugin(workflowSpine, { graph, toolPrefix: 'x', maxIterations: 3 })
    const assembly = await ctx.systemPrompt.assemble()
    const spineSection = assembly.sections.find(s => s.name === 'tool:x-spine')
    expect(spineSection).toBeDefined()
    expect(spineSection!.text).toContain('orchestrator')
    expect(spineSection!.text).toContain('Delegate')
    expect(spineSection!.text).toContain('Never take over')
    // Verify tool prefix is used, not hardcoded 'wf_'
    expect(spineSection!.text).toContain('x_edit')
    expect(spineSection!.text).toContain('x_redo')
    expect(spineSection!.text).toContain('review nodes')
    expect(spineSection!.text).toContain('not a reason to skip')
    expect(spineSection!.text).toContain('Never claim, restate, or re-apply a person')
    expect(spineSection!.text).toContain('user gate is answered only by the person')
    expect(spineSection!.text).toContain('graph-control skip')
    expect(spineSection!.text).toContain('x_edit skip')
    expect(spineSection!.text).not.toContain('apply user steering')
    expect(spineSection!.text).not.toContain('skip the tests')
  })

  it('headless CEO cannot write user-relayed (downgraded to ceo)', async () => {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(UserQuestionService)
    ctx.on('user-questions/request', request => Promise.resolve({
      answers: request.questions.map(question => ({ id: question.id, selected: ['proceed'] })),
    }))
    // Mount with personAttached: false and interactiveOpen: false (headless)
    await ctx.plugin(workflowSpine, {
      graph, toolPrefix: 'wf', maxIterations: 3,
      personAttached: false, interactiveOpen: false,
    })
    const assembly = await ctx.systemPrompt.assemble()
    const spineSection = assembly.sections.find(s => s.name === 'tool:wf-spine')
    expect(spineSection).toBeDefined()
    // The headless CEO should not see the relay sentence
    expect(spineSection!.text).not.toContain("source to 'user-relayed'")
  })

  it('GUI CEO preset counts as attached via interactiveOpen', async () => {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(UserQuestionService)
    ctx.on('user-questions/request', request => Promise.resolve({
      answers: request.questions.map(question => ({ id: question.id, selected: ['proceed'] })),
    }))
    // Mount with interactiveOpen: true and personAttached: false (as ceo.patch.yml would)
    await ctx.plugin(workflowSpine, {
      graph, toolPrefix: 'wf', maxIterations: 3,
      interactiveOpen: true, personAttached: false,
      openCommand: '/bin/true', steerCommand: '/bin/true', steerArgs: [],
    })
    const assembly = await ctx.systemPrompt.assemble()
    const spineSection = assembly.sections.find(s => s.name === 'tool:wf-spine')
    expect(spineSection).toBeDefined()
    // The GUI CEO should see the relay sentence because effectivePersonAttached is true
    expect(spineSection!.text).toContain("source to 'user-relayed'")
  })

  it('skip provenance via wf_edit records source ceo', async () => {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    const skipGraph: WorkflowGraph = {
      name: 't',
      start_node: 'a',
      nodes: {
        a: { _type: 'AgentNode', id: 'a', role: 'builder', prompt_template: 'A', writes: [], reads: [] },
        b: { _type: 'AgentNode', id: 'b', role: 'builder', prompt_template: 'B', writes: [], reads: [] },
      },
      edges: [{ source: 'a', target: 'b' }],
    }
    await ctx.plugin(workflowSpine, { graph: skipGraph, toolPrefix: 'wf', maxIterations: 3 })
    await run(ctx, 'wf_edit', { node: 'b', action: 'skip' })
    const state = await run(ctx, 'wf_state', {}) as any
    expect(state.skip_provenance).toBeDefined()
    expect(state.skip_provenance.b).toBeDefined()
    expect(state.skip_provenance.b.source).toBe('ceo')
    expect(state.skipped).toEqual(['b'])
  })
})

/** A graph whose first two nodes are commands, so the plugin can run them. */
const commandGraph: WorkflowGraph = {
  name: 'commands',
  start_node: 'prepare',
  nodes: {
    prepare: {
      _type: 'FnNode', id: 'prepare', command: 'printf prepared',
      reads: [], writes: ['note.txt'],
    },
    stop: {
      _type: 'GateNode', id: 'stop', evaluator_type: 'fn', evaluator_command: 'echo proceed',
      reads: [], writes: [],
    },
    work: {
      _type: 'AgentNode', id: 'work', role: 'builder', prompt_template: 'W', reads: [], writes: [],
    },
  },
  edges: [
    { source: 'prepare', target: 'stop' },
    { source: 'stop', target: 'work', condition: 'proceed' },
    { source: 'stop', target: 'prepare', condition: 'reloop' },
  ],
}

/** A shell executor that really runs the command, for the composition test. */
function realShell(): object {
  return {
    resolve: (request: { command: string; workdir?: string; timeoutMs?: number; env?: Record<string, string> }) => ({
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: request.timeoutMs ?? 10_000,
      onExpiry: 'kill',
      stdoutMaxBytes: 100_000,
      env: request.env,
      sandboxPolicy: undefined,
    }),
    execute: async (spec: { command: string; workdir: string; timeoutMs: number; env?: Record<string, string> }) => ({
      result: async () => {
        const started = spawnSync('/bin/sh', ['-c', spec.command], {
          cwd: spec.workdir, encoding: 'utf8', timeout: spec.timeoutMs,
          env: { ...process.env, ...spec.env },
        })
        return {
          exitCode: started.status, signal: started.signal, timedOut: false, aborted: false,
          timeoutMs: spec.timeoutMs,
          stdout: { text: started.stdout ?? '', truncated: false },
          stderr: { text: started.stderr ?? '', truncated: false },
        }
      },
    }),
  }
}

describe('deterministic nodes run without the driver', () => {
  it.each([true, false])('classifies an agent with dispatch=%s for the configured driver', async (dispatched) => {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    ctx.provide('shell', realShell() as never)
    await ctx.plugin(workflowSpine, {
      graph: {
        name: 'worker', start_node: 'work', edges: [],
        nodes: {
          work: { _type: 'AgentNode', id: 'work', prompt_template: 'Do work',
            ...(dispatched ? { dispatch_command: 'true' } : {}) },
        },
      },
      executeCommands: true,
    })
    const configured = ctx.get(workflowSpine.CONFIGURED_RUN_SERVICE) as workflowSpine.ConfiguredRun
    expect(configured.automatic).toBe(dispatched)
    if (dispatched) expect(await configured.advance()).toMatchObject({ status: 'done' })
  })

  async function commandSetup(options: { executeCommands: boolean }): Promise<Context> {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    ctx.provide('shell', realShell() as never)
    await ctx.plugin(workflowSpine, {
      graph: commandGraph, toolPrefix: 'wf', maxIterations: 3, ...options,
    })
    return ctx
  }

  it('runs the FnNode and the fn gate, then hands over only the agent node', async () => {
    const ctx = await commandSetup({ executeCommands: true })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({
      status: 'ready', nodes: [{ id: 'work', _type: 'AgentNode' }],
    })
    const state = (await run(ctx, 'wf_state', {})) as { completed: string[]; gate_verdicts: Record<string, string> }
    expect(state.completed).toContain('prepare')
    expect(state.completed).toContain('stop')
    expect(state.gate_verdicts).toEqual({ stop: 'proceed' })
  })

  it('hands the deterministic nodes over when execution is off', async () => {
    const ctx = await commandSetup({ executeCommands: false })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'prepare' }] })
  })

  it('requires a verdict for a gate with a command', async () => {
    const ctx = await commandSetup({ executeCommands: false })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'prepare' }] })
    await run(ctx, 'wf_done', { node: 'prepare' })
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'ready', nodes: [{ id: 'stop' }] })
    await expect(run(ctx, 'wf_verdict', { gate: 'stop' }))
      .rejects.toThrow(/routes on a verdict/)
    expect(await run(ctx, 'wf_verdict', { gate: 'stop', verdict: 'proceed' }))
      .toMatchObject({ status: 'routed', gate: 'stop', verdict: 'proceed' })
  })

  it('reports a failing command and does not advance the cursor', async () => {
    const ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    ctx.provide('shell', realShell() as never)
    await ctx.plugin(workflowSpine, {
      graph: {
        ...commandGraph,
        nodes: {
          ...commandGraph.nodes,
          prepare: { _type: 'FnNode', id: 'prepare', command: 'echo broken >&2; exit 4', reads: [], writes: [] },
        },
      },
      toolPrefix: 'wf',
      executeCommands: true,
    })

    const result = (await run(ctx, 'wf_next', {})) as { status: string; exit_code: number; stderr: string }
    expect(result.status).toBe('failed')
    expect(result.exit_code).toBe(4)
    expect(result.stderr).toBe('broken')
    expect(await run(ctx, 'wf_next', {})).toMatchObject({ status: 'failed' })
  })
})


describe('spine threads custom field as feedback', () => {
  it('passes custom text from a selected verdict as feedback to the gate bridge', async () => {
    // When user-questions returns a selected verdict plus custom text (the note),
    // the spine records the custom text as feedback via the gate bridge.
    const tmp = mkdtempSync(join(tmpdir(), 'spine-custom-fb-'))
    try {
      const { ctx } = await gateRun(graphWithReads, tmp,
        question => Promise.resolve({
          answers: [{ id: question.id, selected: ['proceed'], custom: 'approved with conditions' }],
        }))
      const result = await run(ctx, 'wf_verdict', { gate: 'gate' }) as { status: string; feedback?: string }
      expect(result.status).toBe('routed')
      expect(result.feedback).toBe('approved with conditions')
    } finally { rmSync(tmp, { recursive: true, force: true }) }
  })
})

describe('recovery checkpoint ownership', () => {
  it('refuses dispatch when the execution checkpoint cannot be replaced', async () => {
    const root = mkdtempSync(join(tmpdir(), 'spine-checkpoint-'))
    try {
      mkdirSync(join(root, '.refactory', 'spine-checkpoint.json'), { recursive: true })
      const ctx = new Context()
      await ctx.plugin(SystemPrompt)
      await ctx.plugin(ToolRuntime)
      let calls = 0
      ctx.provide('shell', { ...realShell(), execute: () => { calls++; throw new Error('must not dispatch') } } as never)
      await ctx.plugin(workflowSpine, { graph: commandGraph, executeCommands: true, workdir: root })
      await expect(run(ctx, 'wf_next', {})).rejects.toThrow(/recovery checkpoint/)
      expect(calls).toBe(0)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})
