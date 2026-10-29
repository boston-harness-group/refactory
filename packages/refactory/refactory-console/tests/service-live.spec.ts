/**
 * The service against the real loop: a real console process, the real registry,
 * the real checkout's modes.
 *
 * The unit spec covers the mapping and the refusals; this covers what happens
 * when the argv the service builds is executed — which is the only way to know
 * the two halves of the process boundary still agree. It reads the same console
 * a person pressing "pick a mode" in the panel reads.
 */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { remoteErrorOf } from '@deepseek-ai/dsh-typert-protocol'
import { RefactoryConsole } from '../src/index.ts'
import type { Config } from '../src/index.ts'

/** The checkout that holds `refactory/` and `.venv/`. */
const REPO = join(import.meta.dirname, '..', '..', '..', '..')

const CONFIG: Config = {
  repoRoot: REPO,
  pythonBin: join(REPO, '.venv', 'bin'),
  timeoutMs: 120_000,
  collectBytes: 8 * 1024 * 1024,
}

/** One context with the real subprocess provider, and the service over it. */
async function setup(): Promise<{ ctx: Context; service: RefactoryConsole }> {
  const ctx = new Context()
  await ctx.plugin(LocalSubprocessRuntime)
  // The Gateway's discovery service is not what this spec is about: the
  // service's own registration is asserted below, and nothing else reads it.
  ctx.provide('typert', {} as never)
  await ctx.plugin(RefactoryConsole, CONFIG)
  return { ctx, service: ctx.refactoryConsole }
}

/** One context with the real provider and the service, overridden settings and all. */
async function setupWith(overrides: Partial<Config>): Promise<RefactoryConsole> {
  const ctx = new Context()
  await ctx.plugin(LocalSubprocessRuntime)
  ctx.provide('typert', {} as never)
  await ctx.plugin(RefactoryConsole, { ...CONFIG, ...overrides })
  return ctx.refactoryConsole
}

const signal = (): AbortSignal => new AbortController().signal

/** The failure a Remote error carries, or a defect when it is not one. */
async function failureOf(operation: Promise<unknown>): Promise<{ code: string; message: string }> {
  try {
    await operation
  } catch (error: unknown) {
    const failure = remoteErrorOf(error)
    if (failure === undefined) throw error
    return { code: failure.code, message: failure.message }
  }
  throw new Error('expected the operation to fail')
}

describe('the picker service against the real loop', () => {
  it('registers itself as the refactoryConsole service when it is plugged', async () => {
    const { ctx } = await setup()
    // `ctx.get`, not `ctx.refactoryConsole`: the property proxy is reserved for
    // declared injections, and this spec asserts the registration itself.
    expect(ctx.get('refactoryConsole')).toBeInstanceOf(RefactoryConsole)
  }, 60_000)

  it('offers the modes the checkout can actually run, each with the verb that runs it', async () => {
    const { service } = await setup()
    const modes = await service.modes(signal())
    const ids = modes.map(mode => mode.id)
    expect(ids).toContain('circle_packing/adaptive')
    const scored = modes.find(mode => mode.id === 'circle_packing/adaptive')
    expect(scored?.scored).toBe(true)
    expect(scored?.action).toBe('start')
    expect(scored?.enabled).toBe(true)
    // The authoring modes are offered too, and each names its own verb rather
    // than the one the loop would refuse. The fast variants are authoring modes as
    // well: `run` for `create_fast` selects nothing, which is the bug this catches.
    const authoring = modes.filter(mode => mode.id.startsWith('default_pack/'))
    expect(authoring.length).toBeGreaterThan(0)
    expect(authoring.every(mode => mode.action === mode.id.split('/')[1])).toBe(true)
    // Any other unscored mode has no verb of its own, so it is run once.
    expect(modes.find(mode => mode.id === 'fix_pack/make_pass')?.action).toBe('run')
  }, 120_000)

  it('answers with every declared knob of a mode, as a panel draws them', async () => {
    const { service } = await setup()
    const modes = await service.modes(signal())
    const knobs = modes.find(mode => mode.id === 'circle_packing/adaptive')?.parameters ?? {}
    expect(Object.keys(knobs).length).toBeGreaterThan(0)
    // A frozen knob is still one a person running the mode may set, which is the
    // whole reason the panel draws it.
    expect(knobs.seed).toMatchObject({ kind: 'int', frozen: true })
    expect(knobs.seed?.description.length).toBeGreaterThan(0)
    expect(knobs.acceptance_mode?.values).toEqual(['strict', 'lenient', 'off'])
    // An enum over numbers is the same choice over the text the console reads back.
    expect(knobs.grid?.values).toEqual(['6', '10', '14', '18', '24'])
    expect(Object.values(knobs).every(knob => ['int', 'float', 'bool', 'str', 'enum'].includes(knob.kind))).toBe(true)
  }, 120_000)

  it('names the outer loops, optimizers and evaluators a run can choose between', async () => {    const { service } = await setup()
    const strategies = await service.strategies(signal())
    expect(strategies.optimizers.map(entry => entry.name)).toContain('random')
    expect(strategies.outer_loops.length).toBeGreaterThan(0)
    expect(strategies.evaluators.length).toBeGreaterThan(0)
    expect(strategies.optimizers.find(entry => entry.name === 'random')?.description.length).toBeGreaterThan(0)
  }, 120_000)

  it('lists the checkout\'s packages of modes with the modes each declares', async () => {
    const { service } = await setup()
    const packages = await service.packages(signal())
    const names = packages.map(pkg => pkg.name)
    expect(names).toContain('circle_packing')
    expect(names).toEqual([...names].sort((left, right) => left.localeCompare(right)))
    expect(packages.find(pkg => pkg.name === 'circle_packing')?.enabled).toBe(true)
    expect(packages.find(pkg => pkg.name === 'circle_packing')?.modes.length).toBeGreaterThan(0)
  }, 120_000)

  it('refuses to start an authoring mode, naming the action that runs it', async () => {
    const { service } = await setup()
    const modes = await service.modes(signal())
    const authoring = modes.find(mode => !mode.scored && mode.action !== 'start')
    expect(authoring, 'the checkout must declare at least one authoring mode').toBeDefined()
    const failure = await failureOf(service.start({ mode: authoring!.id }, signal()))
    expect(failure.code).toBe('refactory/authoring-mode')
    expect(failure.message).toContain(`"${authoring!.action}" action`)
  }, 120_000)

  it('reports a run that is not there as the console\'s own refusal', async () => {
    const { service } = await setup()
    const failure = await failureOf(service.status(join(REPO, '.factory', 'loop', 'no-such-run'), 0, signal()))
    expect(failure.code).toBe('refactory/console-failed')
    // The console resolves an absent address itself and names the places it
    // looked, rather than reporting it as a directory that is not a run.
    expect(failure.message).toContain('does not exist (checked')
  }, 120_000)

  it('pages a real journal through the console the service builds argv for', async () => {
    // T-14/T-15 against the real parser: the two activity flags the service now
    // always sends must be options this console accepts, and the answer they
    // produce must be the page the next poll continues from. A console that
    // refused them would answer through the one-shot fallback, so the answer's own
    // `after`/`page_size`/`total` is what proves it understood the flags.
    const { service } = await setup()
    const run = mkdtempSync(join(tmpdir(), 'refactory-page-'))
    try {
      mkdirSync(join(run, '.refactory'), { recursive: true })
      writeFileSync(join(run, '.refactory', 'events.jsonl'), Array.from({ length: 250 }, (_, index) =>
        JSON.stringify({ schema_version: 1, run_id: run, invocation: 'inv-1', kind: 'worker_event', i: index }) + '\n')
        .join(''))
      const frame = await service.status(run, 200, signal())
      expect(frame.activity).toMatchObject({ after: 200, page_size: 200, total: 250, has_more: false })
      expect(frame.activity?.events).toHaveLength(50)
      expect(frame.activity?.cursor).toBe(250)
    } finally {
      rmSync(run, { recursive: true, force: true })
    }
  }, 120_000)

  it('steers a real run: a note is read back, and a `user` gate reports its verdicts', async () => {
    const { service } = await setup()
    const run = mkdtempSync(join(tmpdir(), 'refactory-steer-'))
    try {
      mkdirSync(join(run, '.refactory'), { recursive: true })
      mkdirSync(join(run, 'graphs'), { recursive: true })
      writeFileSync(
        join(run, '.refactory', 'spine-status.json'),
        JSON.stringify({ status: 'ready', nodes: ['approve'], completed: ['draft'] }),
      )
      writeFileSync(join(run, 'graphs', 'p-review.graph.json'), JSON.stringify({
        nodes: { approve: { _type: 'GateNode', evaluator_type: 'user', gate_prompt: 'approve the draft' } },
        edges: [
          { source: 'approve', target: 'publish', condition: 'proceed' },
          { source: 'approve', target: 'reject', condition: 'halt' },
        ],
      }))

      const read = await service.steer({ run }, signal())
      expect(read.exists).toBe(false)
      expect(read.gates).toEqual([
        { id: 'approve', prompt: 'approve the draft', verdicts: ['proceed', 'halt'] },
      ])

      const written = await service.steer({ run, note: 'keep it short', kind: 'constraint' }, signal())
      expect(written.entries.at(-1)).toMatchObject({ kind: 'constraint', text: 'keep it short' })
      expect(written.constraints).toEqual(['keep it short'])

      const answered = await service.steer({ run, verdicts: ['approve=proceed'] }, signal())
      expect(answered.verdicts).toEqual({ approve: 'proceed' })
    } finally {
      rmSync(run, { recursive: true, force: true })
    }
  }, 120_000)
  // §11 item 7 / T-14 against the real process: the console's answer is larger
  // than the collection cap. The unit spec proves the verdict over a fake reader;
  // this proves the real subprocess seam reports it, which is the failure the
  // reported issue actually hit — a capped payload used to surface as "the console
  // printed no JSON", indistinguishable from a console that printed nothing.
  it('names a real capped payload as truncation, not as missing JSON', async () => {
    const root = mkdtempSync(join(tmpdir(), 'refactory-cap-'))
    const run = join(root, 'run')
    mkdirSync(join(run, '.refactory'), { recursive: true })
    const records = Array.from({ length: 400 }, (_, i) => JSON.stringify({
      schema_version: 1, at: '2026-10-26T09:00:00+00:00', kind: 'worker',
      node_id: `n${i}`, summary: 'z'.repeat(400),
    }))
    writeFileSync(join(run, '.refactory', 'events.jsonl'), records.join('\n') + '\n')
    writeFileSync(join(run, '.refactory', 'invocation.json'), JSON.stringify({ id: 'inv-1' }))
    try {
      // 2 KiB cannot hold this run's answer, whatever the console emits.
      const service = await setupWith({ collectBytes: 2048 })
      let caught: { code: string; message: string; details: unknown } | undefined
      try {
        await service.status(run, 0, signal())
      } catch (error: unknown) {
        caught = remoteErrorOf(error) as typeof caught
      }
      expect(caught?.code).toBe('refactory/console-truncated')
      expect(caught?.message).not.toMatch(/printed no JSON/)
      expect(caught?.message).toMatch(/2048/)
      expect(caught?.details).toMatchObject({ capBytes: 2048, retainedBytes: 2048 })
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  }, 120_000)
})
