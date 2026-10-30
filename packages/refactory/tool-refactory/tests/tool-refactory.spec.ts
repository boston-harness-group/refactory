import type { Duplex } from 'node:stream'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { SubprocessRuntime } from '@deepseek-ai/dsh-subprocess'
import type {
  SubprocessCollectedOutputs,
  SubprocessHandle,
  SubprocessOutcome,
  SubprocessOutputRead,
  SubprocessSpawnSpec,
  SubprocessTerminalEnvironment,
} from '@deepseek-ai/dsh-subprocess'

import * as tool from '../src/index.ts'
import { presentCall, presentResult, presentationMeta } from '../src/present.ts'
import { CHAT } from '../src/generated-help.ts'
import { CONSOLE_DEFAULTS, Console } from '../src/console.ts'

/**
 * Drives the real plugin body over a scriptable subprocess service: the tool
 * registration, its argument validation, and the exact argv it hands the
 * console. The console is not run here — `refactory/tests/test_gui_loop.py`
 * covers it against the real loop, and this covers the boundary it is called
 * across.
 */

function runResult(stdout: string, exitCode = 0, stderr = '', lossy = false): {
  stdout: { text: string; lossy: boolean }
  stderr: { text: string }
  outcome: SubprocessOutcome
} {
  return { stdout: { text: stdout, lossy }, stderr: { text: stderr }, outcome: { exitCode, signal: null } }
}

type Scripted = ReturnType<typeof runResult>

class FakeReader {
  constructor(private readonly read: { text: string; lossy?: boolean }) {}
  readFrom(_fromByte: number): SubprocessOutputRead {
    return { text: this.read.text, nextOffset: this.read.text.length, lossy: this.read.lossy === true }
  }
}

class FakeHandle implements SubprocessHandle {
  readonly stdin = undefined
  readonly stdout = undefined
  readonly stderr = undefined
  readonly control: Duplex | undefined = undefined
  readonly collected: SubprocessCollectedOutputs
  readonly done: Promise<SubprocessOutcome>
  constructor(spec: SubprocessSpawnSpec, scripted: Scripted, dropReaders = false) {
    spec.signal?.addEventListener('abort', () => { this.terminated = true }, { once: true })
    this.collected = dropReaders ? {} : {
      stdout: new FakeReader(scripted.stdout),
      stderr: new FakeReader(scripted.stderr),
    }
    this.done = Promise.resolve(scripted.outcome)
  }
  terminated = false
  terminate(): void { this.terminated = true }
  waitForExit(): Promise<boolean> { return Promise.resolve(true) }
}

class FakeSubprocess extends SubprocessRuntime {
  spawns: SubprocessSpawnSpec[] = []
  handler: () => Scripted = () => runResult('{}')
  dropReaders = false
  override resolveExecutable(command: string): Promise<string> { return Promise.resolve(command) }
  override terminalEnvironment(): Promise<SubprocessTerminalEnvironment> {
    return Promise.resolve({ platform: process.platform === 'win32' ? 'windows' : 'posix' })
  }
  override spawnTerminal(): Promise<never> { throw new Error('the refactory tool spawns collectors, never terminals') }
  override spawn(spec: SubprocessSpawnSpec): SubprocessHandle {
    this.spawns.push(spec)
    return new FakeHandle(spec, this.handler(), this.dropReaders)
  }
}

const CONFIG = {
  toolName: 'refactory',
  repoRoot: '/repo',
  pythonBin: '/repo/.venv/bin',
  timeoutMs: 60_000,
  collectBytes: 1_048_576,
}

async function setup(config: Partial<tool.Config> = {}) {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(FakeSubprocess)
  const subprocess = ctx.subprocess as FakeSubprocess
  await ctx.plugin(tool, { ...CONFIG, ...config })
  return { ctx, subprocess }
}

let counter = 0
function call(ctx: Context, args: unknown) {
  return ctx.tools.execute({
    signal: new AbortController().signal,
    callId: ToolCallId(`call-${++counter}`),
    name: 'refactory',
    arguments: args,
  })
}

function text(result: { content: { type: string; text?: string }[] }): string {
  return result.content.filter(block => block.type === 'text').map(block => block.text).join('')
}

/** The assembled system prompt, which is where a registered section lands. */
async function promptText(ctx: Context): Promise<string> {
  const assembly = await ctx.systemPrompt.assemble()
  return assembly.sections.map(section => typeof section.text === 'string' ? section.text : '').join('\n')
}

describe('dsh-tool-refactory', () => {
  it('registers one `refactory` tool with an action enum', async () => {
    const { ctx } = await setup()
    const schema = ctx.tools.schemas().find(entry => entry.name === 'refactory')
    expect(schema).toBeDefined()
    const action = (schema!.parameters as { properties: Record<string, { enum?: string[] }> }).properties.action
    expect(action?.enum).toEqual(Object.keys(CHAT))
    expect(action?.enum).toEqual(expect.arrayContaining(['help', 'docs', 'workers', 'profiles', 'profile-resolve']))
  })

  it('runs the console through the subprocess capability with the configured interpreter', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify([{ id: 'circle_packing/adaptive', startable: true }]))
    const result = await call(ctx, { action: 'modes' })
    expect(subprocess.spawns).toHaveLength(1)
    expect(subprocess.spawns[0]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'modes',
    ])
    expect(subprocess.spawns[0]!.cwd).toBe('/repo')
    expect(text(result)).toContain('circle_packing/adaptive')
  })

  it('defaults the interpreter against the repo root rather than the process cwd', async () => {
    const { ctx, subprocess } = await setup({ pythonBin: '', repoRoot: '/elsewhere' })
    subprocess.handler = () => runResult('[]')
    await call(ctx, { action: 'modes' })
    expect(subprocess.spawns[0]!.argv[0]).toBe('/elsewhere/.venv/bin/python')
  })

  it('passes each action through as console flags and positionals', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('{}')
    await call(ctx, {
      action: 'start',
      mode: 'circle_packing/adaptive',
      runId: 'r1',
      runsRoot: '/runs',
      settings: { epochs: 2, batch_size: 3, seed: 7 },
    })
    expect(subprocess.spawns[0]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'start',
      '--run-id', 'r1', '--runs-root', '/runs',
      '--epochs', '2', '--batch-size', '3', '--seed', '7',
      'circle_packing/adaptive',
    ])
  })

  it('spells snake_case settings as the console\'s kebab-case flags', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('{}')
    await call(ctx, { action: 'start', mode: 'm/n', settings: { max_evaluations: 4 } })
    expect(subprocess.spawns[0]!.argv).toContain('--max-evaluations')
  })

  it('names the candidate as a positional after the run scope', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('{}')
    await call(ctx, { action: 'candidate', run: 'r1', candidate: 'abc' })
    expect(subprocess.spawns[0]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'candidate',
      '--run', 'r1', 'abc',
    ])
  })

  it('passes a false boolean as no flag and a true one as a bare flag', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('{}')
    await call(ctx, { action: 'artifacts', run: 'r1', contents: false })
    expect(subprocess.spawns[0]!.argv).not.toContain('--contents')
    await call(ctx, { action: 'artifacts', run: 'r1', contents: true })
    expect(subprocess.spawns[1]!.argv).toContain('--contents')
  })

  it('refuses an action missing the thing that addresses it, before spawning', async () => {
    const { ctx, subprocess } = await setup()
    expect(text(await call(ctx, { action: 'status' }))).toContain('needs `run`')
    expect(text(await call(ctx, { action: 'start' }))).toContain('needs `mode`')
    expect(text(await call(ctx, { action: 'candidate', run: 'r1' }))).toContain('`candidate`')
    expect(subprocess.spawns).toHaveLength(0)
  })

  it('surfaces the console\'s own refusal as a tool error', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ error: '/nope is not a run directory' }), 2)
    const result = await call(ctx, { action: 'status', run: '/nope' })
    expect(result.isError).toBe(true)
    expect(text(result)).toContain('/nope is not a run directory')
  })

  it('reports a console that printed no JSON instead of returning nothing', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('not json')
    expect(text(await call(ctx, { action: 'modes' }))).toContain('printed no JSON')
  })

  it('reports a nonzero exit with no JSON and no stderr as a status', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('', 3)
    expect(text(await call(ctx, { action: 'modes' }))).toContain('exited with 3')
  })

  it('reports a provider that returned no collected readers', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.dropReaders = true
    expect(text(await call(ctx, { action: 'modes' }))).toContain('no collected output')
  })

  it('returns the console\'s JSON beside the action that produced it', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({
      run_id: 'r1', state: 'running', counters: { epoch: 2 },
      text: 'Run r1  probe\nState: running; no score yet.\nEpoch 2, generation 0, 0 evaluation(s).',
    }))
    const result = await call(ctx, { action: 'status', run: 'r1' })
    expect(result.isError).toBeFalsy()
    expect(text(result)).toContain('Run r1')
    expect(text(result)).toContain('running')
  })

  it('summarizes every action without printing the whole payload', async () => {
    const payloads: Record<string, unknown> = {
      modes: [{ id: 'a/b', startable: true }, { id: 'c/d', startable: false }],
      runs: [{ run_id: 'r1', mode: 'a/b', best_score: 1.5, finished: true }],
      start: { run_id: 'r1', mode: 'a/b' },
      status: { run_id: 'r1', state: 'finished', best_score: 1.5, counters: { epoch: 1, generation: 2, evaluations: 3 }, text: 'Run r1  \nState: finished; best score 1.5.\nEpoch 1, generation 2, 3 evaluation(s).' },
      artifacts: { artifacts: [{ name: 'report', media_type: 'text/markdown', exists: true }] },
      leaderboard: [{ id: 'x', score: 1.5 }],
      candidate: { candidate: 'x', parent: 'y', rendered: 'Parameters:\n- grid: 18 → 24' },
      stop: { stopped: true },
    }
    const { ctx, subprocess } = await setup()
    for (const [action, payload] of Object.entries(payloads)) {
      subprocess.handler = () => runResult(JSON.stringify(payload))
      const args = { action, run: 'r1', mode: 'a/b', candidate: 'x' }
      const summary = text(await call(ctx, args))
      expect(summary.length, action).toBeGreaterThan(0)
      // The summary is prose about the payload; the payload itself travels
      // structured, for the card, and must not also be dumped into the text.
      if (action === 'modes') expect(summary).toContain(JSON.stringify((payload as unknown[])[0]))
      else expect(summary, action).not.toContain('{"')
    }
  })

  it('says a run was already gone rather than claiming it stopped', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ stopped: false, reason: 'the run is not running' }))
    expect(text(await call(ctx, { action: 'stop', run: 'r1' }))).toContain('the run is not running')
  })

  it('presents the call and the result under the tool tile', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify([{ id: 'a/b', startable: true }]))
    await call(ctx, { action: 'modes' })
    expect(presentCall({ action: 'start', mode: 'a/b' })).toMatchObject({
      card: 'generic', title: 'Start optimizing a/b',
    })
    // The completed card renders the payload `presentationMeta` persisted, which is
    // where the host hands it back; there is no `data` on the completed call itself.
    const view = presentResult({ action: 'modes' }, {
      isError: false, meta: presentationMeta({ action: 'modes', data: [{ id: 'a/b', startable: true }] }),
    }, data => tool.renderResult({ action: 'modes' }, data))
    expect(view).toMatchObject({ card: 'generic', title: 'List the available modes' })
    expect(view?.card === 'generic' ? view.content : []).toEqual(tool.renderResult({ action: 'modes' }, [{ id: 'a/b', startable: true }]))
  })

  it('projects the console payload for the browser card', () => {
    const meta = presentationMeta({ action: 'status', data: { run_id: 'r1', state: 'running' } })
    expect(meta).toEqual({ action: 'status', data: { run_id: 'r1', state: 'running' } })
  })

  it('maps the steering, winner and package actions onto the console verbs', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('{}')
    await call(ctx, { action: 'strategies' })
    expect(subprocess.spawns[0]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'strategies',
    ])
    await call(ctx, { action: 'steer', run: 'r1', note: 'no numpy', kind: 'constraint' })
    expect(subprocess.spawns[1]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'steer',
      '--run', 'r1', '--note', 'no numpy', '--kind', 'constraint', '--source', 'ceo',
    ])
    await call(ctx, { action: 'adopt', run: 'r1', as: 'adopted_demo' })
    expect(subprocess.spawns[2]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'adopt',
      '--run', 'r1', '--as', 'adopted_demo',
    ])
    await call(ctx, { action: 'registry', packages: 'remove', target: 'phase_pack' })
    expect(subprocess.spawns[3]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'registry',
      'remove', 'phase_pack',
    ])
    await call(ctx, { action: 'create', name: 'new_pkg', brief: 'count up' })
    expect(subprocess.spawns[4]!.argv).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'create',
      '--name', 'new_pkg', '--brief', 'count up',
    ])
  })

  it('refuses an adopt with no package name before spawning', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult('{}')
    expect(text(await call(ctx, { action: 'adopt', run: 'r1' }))).toContain('`as`')
    expect(subprocess.spawns).toHaveLength(0)
  })

  it('registers a prompt section telling the model a run outlives the call', async () => {
    const { ctx } = await setup()
    expect(await promptText(ctx)).toContain('can outlive a chat')
  })

  it('honours a renamed tool', async () => {
    const { ctx } = await setup({ toolName: 'loop' })
    expect(ctx.tools.schemas().map(entry => entry.name)).toContain('loop')
    expect(await promptText(ctx)).toContain('Tool: loop.')
  })

  it('presents through the registered definition, so the runtime renders the same view', async () => {
    const { ctx } = await setup()
    const definition = ctx.tools.get('refactory')
    expect(definition?.presentCall?.({ action: 'modes' })).toMatchObject({
      card: 'generic', title: 'List the available modes', kind: 'read',
    })
    expect(definition?.presentResult?.({ action: 'modes' }, {
      content: [{ type: 'text', text: '1 mode(s)' }], isError: false,
      meta: { action: 'modes', data: [{ id: 'a/b', startable: true }] },
    })).toMatchObject({ card: 'generic', title: 'List the available modes', content: [{ type: 'text', text: expect.stringContaining('a/b') }] })
    expect(definition?.presentResult?.({ action: 'modes' }, { content: [], isError: true })).toBeUndefined()
  })
})

describe('invocationFor', () => {
  it('maps best, edit and design onto their console verbs', () => {
    expect(tool.invocationFor({ action: 'best', run: 'r1', state: true })).toMatchObject({
      verb: 'best', positional: [], options: { run: 'r1', state: true },
    })
    expect(tool.invocationFor({
      action: 'edit', mode: 'm/n', name: 'pkg', brief: 'faster', allowRegression: true, attempts: 2,
    })).toMatchObject({
      verb: 'edit',
      options: { mode: 'm/n', name: 'pkg', brief: 'faster', 'allow-regression': true, attempts: 2 },
    })
    expect(tool.invocationFor({
      action: 'design', name: 'pkg', brief: 'a thing', check: 'pytest', attempts: 3,
    })).toMatchObject({
      verb: 'design',
      options: { name: 'pkg', brief: 'a thing', check: 'pytest', attempts: 3 },
    })
  })

  it('defaults the registry verb to list, and refuses authoring calls missing an argument', () => {
    expect(tool.invocationFor({ action: 'registry' })).toMatchObject({
      verb: 'registry', positional: ['list', ''],
    })
    expect(() => tool.invocationFor({ action: 'create', name: 'p' })).toThrow(/needs `name` and `brief`/)
    expect(() => tool.invocationFor({ action: 'edit', name: 'p', brief: 'b' })).toThrow(/needs `mode`, `name` and `brief`/)
    expect(() => tool.invocationFor({ action: 'edit', mode: 'm/n', brief: 'b' })).toThrow(/needs `mode`, `name` and `brief`/)
    expect(() => tool.invocationFor({ action: 'design', brief: 'b', check: 'c' })).toThrow(/needs `name`, `brief` and `check`/)
    expect(() => tool.invocationFor({ action: 'design', name: 'p', check: 'c' })).toThrow(/needs `name`, `brief` and `check`/)
  })

  it('refuses to answer a gate unless the caller says nobody is at the keyboard', async () => {
    // The mistake #92 names: the CEO pre-asked the person, then passed their answer
    // to a headless call, pre-answering a gate the runtime would have put to them.
    // `headless` is the caller saying which case this is, so it cannot be accidental.
    const { ctx, subprocess } = await setup()
    expect(() => tool.invocationFor({
      action: 'create', name: 'p', brief: 'b', approve: ['gate_strategy=proceed'],
    })).toThrow(/needs `headless: true`/)
    expect(text(await call(ctx, {
      action: 'create', name: 'p', brief: 'b', approve: ['gate_strategy=proceed'],
    }))).toContain('wf_open')
    expect(subprocess.spawns).toHaveLength(0)

    // The same call, saying it: the answers reach the console as its own flags.
    const forwarded = tool.invocationFor({
      action: 'create', name: 'p', brief: 'b', approve: ['gate_strategy=proceed'], headless: true,
    })
    expect(forwarded.options.verdict).toEqual(['gate_strategy=proceed'])
    expect(forwarded.options).not.toHaveProperty('headless')
  })

  it('titles the pending call in plain words, naming its subject', () => {
    const title = (args: Parameters<typeof presentCall>[0]): string => {
      const view = presentCall(args)
      return view.card === 'generic' ? view.title : ''
    }
    expect(title({ action: 'modes' })).toBe('List the available modes')
    expect(title({ action: 'status', run: '/home/lab/.factory/loop/r1' })).toBe('Check run r1')
    expect(title({ action: 'candidate', run: 'r1', candidate: 'abc' })).toBe('Read candidate abc of run r1')
    expect(title({ action: 'design', name: 'linecount' })).toBe('Design and build linecount')
    expect(title({ action: 'adopt', run: 'r1', as: 'first_tuned' })).toBe('Adopt the winner of run r1 as first_tuned')
    expect(title({ action: 'steer', run: 'r1', note: 'no numpy', kind: 'constraint' })).toBe('Add a constraint to run r1: "no numpy"')
    expect(title({ action: 'docs', query: 'budgets' })).toBe('Search the docs for "budgets"')
  })

  it('titles every action the tool accepts in plain words', () => {
    for (const action of Object.keys(CHAT)) {
      const view = presentCall({ action })
      expect(view.card === 'generic' ? view.title : '', action).not.toMatch(/^refactory /)
    }
  })

  it('renders a completed call whose result carries no payload', () => {
    expect(presentResult({ action: 'modes' }, { isError: false }, data => tool.renderResult({ action: 'modes' }, data)))
      .toMatchObject({ card: 'generic', title: 'List the available modes', content: [{ type: 'text' }] })
  })
})

describe('renderResult', () => {
  function rendered(action: string, value: unknown): string {
    const blocks = tool.renderResult({ action } as Parameters<typeof tool.renderResult>[0], value)
    return blocks.map(block => block.type === 'text' ? block.text : '').join('')
  }

  it('renders modes, skipping entries that are not records', () => {
    expect(rendered('modes', { not: 'an array' })).toContain('0 mode(s)')
    expect(rendered('modes', [{ id: null, startable: true }, 'nope', { id: { nested: 1 } }]))
      .toContain('- {"nested":1}')
  })

  it('names both routes, the gates that ask a person, and the knobs', () => {
    // The console's own fields, rendered together: a caller shown only `action`
    // picks the headless route and pre-answers the gate the person would have been
    // asked at (#92, #93).
    const text = rendered('modes', [{
      id: 'default_pack/design', action: 'design', scored: false, startable: true,
      interactive: true,
      gates: [{ id: 'gate_research', user: false }, { id: 'gate_strategy', user: true }],
      parameters: { brief: {}, check: {} },
    }])
    expect(text).toContain('1 mode(s):')
    expect(text).toContain('run: design')
    expect(text).toContain('or wf_open to drive it node by node')
    expect(text).toContain('gates: gate_research, gate_strategy (user)')
    expect(text).toContain('knobs: brief, check')
  })

  it('still says what it does not know about a mode', () => {
    expect(rendered('modes', [{ id: 'a/b' }])).toContain('run: unknown')
    expect(rendered('modes', [{ id: 'a/b', action: 'start', startable: false }]))
      .toContain('not startable')
  })

  it('renders runs with the state each one is in', () => {
    expect(rendered('runs', { not: 'an array' })).toContain('0 run(s)')
    expect(rendered('runs', ['nope', { run_id: 'r1', mode: 'a/b' }])).toContain('in flight')
  })

  it('renders the strategy seams, skipping malformed groups and entries', () => {
    expect(rendered('strategies', 'nope')).toContain('0 outer loop(s)')
    expect(rendered('strategies', { outer_loops: 'nope' })).toContain('0 outer loop(s)')
    expect(rendered('strategies', { optimizers: ['nope', { name: 'autoresearch', description: 'searches' }] }))
      .toContain('- autoresearch: searches')
  })

  it('renders status without text as a minimal line', () => {
    expect(rendered('status', { run_id: 'r1', state: 'running' }))
      .toContain('Run r1 is running')
  })

  it('renders status from payload text field', () => {
    const result = rendered('status', {
      run_id: 'test_run', state: 'running',
      text: 'Run test_run  my_mode\nSteps done: 3\nOwner: alive',
    })
    expect(result).toContain('Run test_run  my_mode')
    expect(result).toContain('Steps done: 3')
    expect(result).toContain('Owner: alive')
  })

  it('renders a leaderboard, saying a row with no score scored nothing', () => {
    expect(rendered('leaderboard', 'nope')).toContain('0 candidate(s)')
    expect(rendered('leaderboard', ['nope', { id: 'x' }])).toContain('- x scored nothing')
  })

  it('names the candidate the payload omits as the call asked for, or nothing', () => {
    expect(rendered('candidate', {})).toContain('Candidate  against no parent:')
  })

  it('renders steer from payload text field', () => {
    const result = rendered('steer', {
      run: '/tmp/run', ledger: '/tmp/run/ledger.md',
      text: 'Recorded verdict gate_strategy=proceed.\nNote recorded.',
    })
    expect(result).toContain('Recorded verdict gate_strategy=proceed')
    expect(result).toContain('Note recorded')
  })

  it('renders steer without text as a minimal line', () => {
    expect(rendered('steer', { run: '/tmp/run' })).toContain('Ledger for /tmp/run')
  })

  it('renders the winner and the adoption', () => {
    expect(rendered('best', { candidate: 'x', score: 1.5 })).toContain('The winner is x, scoring 1.5.')
    expect(rendered('best', { candidate: 'x' })).toContain('The winner is x, scoring nothing.')
    expect(rendered('adopt', { mode: 'm/n', adopted_from: 'r1', package_root: '/p', score: 2 }))
      .toContain('The adopted mode scored 2')
    expect(rendered('adopt', { mode: 'm/n', adopted_from: 'r1', package_root: '/p' }))
      .toContain('scored nothing')
  })

  it('reports a forwarded authoring call by what it produced', () => {
    const succeeded = rendered('create', { ok: true, exit_code: 0, result: { output: 'wrote pkg' } })
    expect(succeeded).toContain('create succeeded (exit 0).')
    expect(succeeded).toContain('wrote pkg')
    expect(rendered('edit', { ok: false, exit_code: 1, result: 'nope' })).toContain('edit failed')
  })

  it('says nothing was stopped when the payload names no reason', () => {
    expect(rendered('stop', { stopped: false })).toContain('unknown')
  })
})

describe('Console', () => {
  it('builds argv without a shell, so a mode id is never interpreted', async () => {
    const ctx = new Context()
    await ctx.plugin(FakeSubprocess)
    const console = new Console(ctx.subprocess, {
      repoRoot: '/repo', pythonBin: '/repo/.venv/bin', timeoutMs: 1_000, collectBytes: 1_024,
    })
    expect(console.argv({ verb: 'status', positional: [], options: { run: 'a b; rm -rf /' } })).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'status', '--run', 'a b; rm -rf /',
    ])
  })

  /** A console over a scripted subprocess, whose outcome each case states. */
  async function consoleOver(scripted: Scripted): Promise<{ console: Console; fake: FakeSubprocess }> {
    const ctx = new Context()
    await ctx.plugin(FakeSubprocess)
    const fake = ctx.subprocess as FakeSubprocess
    fake.handler = () => scripted
    const console = new Console(ctx.subprocess, {
      repoRoot: '/repo', pythonBin: '/repo/.venv/bin', timeoutMs: 1_000, collectBytes: 1_024,
    })
    return { console, fake }
  }

  it('decodes JSON when the caller passes no signal of its own', async () => {
    const { console } = await consoleOver(runResult('[1]'))
    await expect(console.run({ verb: 'modes' })).resolves.toMatchObject({ ok: true, value: [1] })
  })

  it('reports the console\'s stderr when it failed without a JSON reason', async () => {
    const { console } = await consoleOver(runResult('', 2, '  broken pipe\n'))
    await expect(console.run({ verb: 'modes' })).resolves.toMatchObject({ ok: false, error: 'broken pipe' })
  })

  it('keeps the payload a failed call printed, which is where a stopped run reports itself', async () => {
    // `console run` writes where the run stopped beside the reason it exited
    // nonzero, so a caller reading that status cannot be handed only the reason.
    const printed = { ok: false, exit_code: 1, error: 'the graph did not finish', run: { state: 'running' } }
    const { console } = await consoleOver(runResult(JSON.stringify(printed), 2))
    await expect(console.run({ verb: 'run', positional: ['p/adaptive'] })).resolves.toMatchObject({
      ok: false,
      error: 'the graph did not finish',
      value: printed,
    })
  })

  it('names the signal when a killed console has no exit code', async () => {
    const killed: Scripted = {
      stdout: { text: '', lossy: false }, stderr: { text: '' }, outcome: { exitCode: null, signal: 'SIGTERM' },
    }
    const { console } = await consoleOver(killed)
    await expect(console.run({ verb: 'modes' })).resolves.toMatchObject({
      ok: false, error: 'exited with SIGTERM',
    })
  })

  it('reports no status at all when neither exit code nor signal is known', async () => {
    const unknown: Scripted = {
      stdout: { text: '', lossy: false }, stderr: { text: '' }, outcome: { exitCode: null, signal: null },
    }
    const { console } = await consoleOver(unknown)
    await expect(console.run({ verb: 'modes' })).resolves.toMatchObject({
      ok: false, error: 'exited with no status',
    })
  })

  it('falls back to stderr when a successful console printed no JSON', async () => {
    const { console } = await consoleOver(runResult('   ', 0, 'warning: nothing to say'))
    await expect(console.run({ verb: 'modes' })).resolves.toMatchObject({
      ok: false, error: 'the console printed no JSON: warning: nothing to say',
    })
  })

  it('reports a payload the collection cap cut as truncation, naming the cap and the retained tail', async () => {
    // T-11 (R-5.1-3, UC-7): the reader's own `lossy` fact decides this, never a
    // guess about the text. The seam delivers the cap and a retained length and
    // no whole-stream total, so the failure names exactly those two numbers plus
    // the argv that reproduces it — and it is not the "printed no JSON" message.
    const tail = '{"activity":{"events":[{"at":"2026-10-26T15:04:05+00:00","kind":"worker_'
    const { console } = await consoleOver(runResult(tail, 0, '', true))
    const result = await console.run({ verb: 'status', options: { run: '/runs/live', after: 0 } })
    expect(result).toMatchObject({
      ok: false,
      truncation: { capBytes: 1_024, retainedBytes: tail.length },
      argv: [
        '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'status', '--run', '/runs/live', '--after', '0',
      ],
    })
    if (result.ok) throw new Error('a truncated payload is not a success')
    expect(result.error).toBe(
      `the console's output was truncated at the 1024-byte collection cap before it printed JSON `
      + `(only ${tail.length} bytes were retained); raise collectBytes or page the read\n`
      + '/repo/.venv/bin/python -m refactory_dsh.console status --run /runs/live --after 0',
    )
    expect(result.error).not.toContain('printed no JSON')
    expect(Object.keys(result.truncation ?? {})).toEqual(['capBytes', 'retainedBytes'])
  })

  it('keeps a complete non-JSON stdout as the no-JSON failure, with no truncation', async () => {
    // T-11's other half (R-5.2, RES-010 / issue 95): a child's own output ahead
    // of the envelope is a complete stream that is not JSON, so it stays the
    // message it always was and gains no truncation verdict.
    const { console } = await consoleOver(runResult('child banner\n{"run_id":"live"}'))
    const result = await console.run({ verb: 'status', options: { run: '/runs/live', after: 0 } })
    expect(result).toMatchObject({
      ok: false,
      error: 'the console printed no JSON: child banner\n{"run_id":"live"}',
    })
    expect(result.ok ? undefined : 'truncation' in result).toBe(false)
  })

  it('treats a lossy read whose retained tail still decodes as the answer it is', async () => {
    // T-12 (R-5.3): the contract does not invent a failure the seam does not
    // prove. The complete payload arrived in the retained tail, so the call is a
    // success with no truncation — saying otherwise would report a broken read
    // for an answer the caller can use.
    const { console } = await consoleOver(runResult('{"run_id":"live","state":"running"}', 0, 'dropped head', true))
    await expect(console.run({ verb: 'status', options: { run: '/runs/live', after: 0 } })).resolves.toEqual({
      ok: true,
      argv: ['/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'status', '--run', '/runs/live', '--after', '0'],
      value: { run_id: 'live', state: 'running' },
    })
  })

  it('drops the paging options once for a console that refuses them, and only for that refusal', async () => {
    // T-13's boundary half (R-6.1): a console older than the paging contract
    // exits 2 with `unrecognized arguments`. Retrying once without those flags —
    // and naming them — keeps the run readable; looping on the refusal would not.
    const { console, fake } = await consoleOver(runResult(''))
    const answers = [
      runResult('', 2, 'console: error: unrecognized arguments: --activity-after 0 --activity-limit 200'),
      runResult('{"run_id":"live"}'),
    ]
    fake.handler = () => answers.shift() ?? runResult('{"run_id":"live"}')
    await expect(console.run({
      verb: 'status',
      options: { run: '/runs/live', after: 0, 'activity-after': 0, 'activity-limit': 200 },
    })).resolves.toEqual({
      ok: true,
      argv: ['/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'status', '--run', '/runs/live', '--after', '0'],
      value: { run_id: 'live' },
      suppressed: ['activity-after', 'activity-limit'],
    })
    expect(fake.spawns).toHaveLength(2)

    // The same refusal on the retry is reported rather than retried again: one
    // attempt per call, never a loop.
    const { console: stubborn, fake: stubbornFake } = await consoleOver(
      runResult('', 2, 'console: error: unrecognized arguments: --activity-after 0'),
    )
    await expect(stubborn.run({
      verb: 'status', options: { run: '/runs/live', 'activity-after': 0 },
    })).resolves.toMatchObject({
      ok: false, error: 'console: error: unrecognized arguments: --activity-after 0',
    })
    expect(stubbornFake.spawns).toHaveLength(2)
  })
})


it('passes a creation\'s data-node and task directives through to the console', async () => {
  // create-v2's `--data-node` and `--task` are what make a dataset mode reachable
  // from the conversation; the console checks both after the run.
  const { ctx, subprocess } = await setup()
  subprocess.handler = () => runResult('{}')
  await call(ctx, { action: 'create', name: 'pkg', brief: 'count', dataNode: true, task: 'probe:Counting' })
  expect(subprocess.spawns[0]!.argv).toEqual([
    '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'create',
    '--name', 'pkg', '--brief', 'count', '--data-node', '--task', 'probe:Counting',
  ])
})

it('tells the model which action runs which kind of mode', async () => {
  // The description is the first thing the model reads; a caller that has to fail
  // `start` to learn that `create` is an authoring mode is a caller the tool could
  // have directed.
  const { ctx } = await setup()
  const definition = ctx.tools.get('refactory')
  // Both routes, and who a gate belongs to: a caller told only the headless verb
  // picks it and pre-answers a gate the person would have been asked at (#92).
  expect(definition?.description).toContain('start optimizes a scored mode')
  expect(definition?.description).toContain('use wf_open')
  expect(await promptText(ctx)).toContain('never invent their verdicts')
  expect(await promptText(ctx)).toContain('headless: true')
})

describe('authoring from the Web GUI', () => {
  it('forwards a person\'s gate answers to the authoring run', async () => {
    // Without a verdict a headless authoring run refuses `gate_strategy`, so every
    // creation from the GUI failed with exit 2 and no way to answer.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, {
      action: 'create',
      name: 'p',
      brief: 'count',
      approve: ['gate_strategy=proceed', 'gate_qa=proceed'],
      headless: true,
    })
    const argv = subprocess.spawns[0]!.argv
    expect(argv).toContain('--verdict')
    expect(argv.filter(part => part === '--verdict')).toHaveLength(2)
    expect(argv).toContain('gate_strategy=proceed')
    expect(argv).toContain('gate_qa=proceed')

    await call(ctx, { action: 'edit', mode: 'counting_pack/fixed_step', name: 'p', brief: 'x', approve: ['gate_strategy=proceed'], headless: true })
    expect(subprocess.spawns[1]!.argv).toContain('gate_strategy=proceed')

    await call(ctx, { action: 'design', name: 'a', brief: 'x', check: 'true', approve: ['gate_strategy=proceed'], headless: true })
    expect(subprocess.spawns[2]!.argv).toContain('gate_strategy=proceed')
  })

  it('shows the reason a failed authoring call gave, stderr included', async () => {
    // `create failed (exit 2)` with an empty payload was all the model or the user
    // saw: the entry point writes its refusal to stderr and the console dropped it.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({
      verb: 'create',
      exit_code: 2,
      ok: false,
      result: { error: 'the strategy gate is answered by a person' },
      stderr: '{"error": "the strategy gate is answered by a person"}',
    }))
    const rendered = text(await call(ctx, { action: 'create', name: 'p', brief: 'count' }))
    expect(rendered).toContain('the strategy gate is answered by a person')
  })

  it('reports an authoring run\'s own progress from status via text field', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({
      run_id: 'demo_pack',
      state: 'running',
      counters: { epoch: 0, generation: 0, evaluations: 0 },
      text: 'Run demo_pack  demo_pack\nPosition: strategy_director (agent)\nSteps done: 3',
      authoring: {
        spine: 'ready',
        completed: ['prepare', 'survey', 'research_director'],
        nodes_due: ['strategy_director'],
        name: 'demo_pack',
        validation: '(no validation has run yet)',
      },
    }))
    const rendered = text(await call(ctx, { action: 'status', run: '.factory/create/demo_pack' }))
    expect(rendered).toContain('demo_pack')
    expect(rendered).toContain('Steps done: 3')
    expect(rendered).toContain('strategy_director')
  })

  it('gives an authoring call long enough to finish', () => {
    // Measured, not guessed: a real creation on this checkout took 48 minutes (37
    // inner runs, one overwatch reloop), so the ceiling has to clear that with room
    // for one more reloop.
    expect(CONSOLE_DEFAULTS.timeoutMs).toBeGreaterThanOrEqual(90 * 60 * 1000)
  })
})

describe('the brief\'s declared inputs', () => {
  it('forwards each input the caller names to the creation', async () => {
    // The validate gate refuses a mode with no way to receive an input the brief
    // stated, so the directive has to survive the hop.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, {
      action: 'create',
      name: 'buildroot_pack',
      brief: 'match a wheel against a package',
      approve: ['gate_strategy=proceed'],
      headless: true,
      inputs: ['package', 'reference_wheel'],
    })
    const argv = subprocess.spawns[0]!.argv
    expect(argv.filter(part => part === '--input')).toHaveLength(2)
    expect(argv).toContain('package')
    expect(argv).toContain('reference_wheel')
  })
})

describe('creating a harness from the conversation', () => {
  it('forwards the creation directive, so a one-off is askable at all', async () => {
    // `create` refuses a request that names neither a class of work nor the one-off
    // special case, and the tool is the third caller beside the CLI and the console:
    // without this argument the only creation reachable from the conversation is one
    // bound to a Task.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, {
      action: 'create',
      name: 'haiku_pack',
      brief: 'one agent drafts a haiku, one op counts syllables',
      singleShot: true,
    })
    const argv = subprocess.spawns[0]!.argv
    expect(argv.slice(0, 4)).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'create',
    ])
    expect(argv).toContain('--single')

    subprocess.spawns.length = 0
    await call(ctx, {
      action: 'create',
      name: 'my_harness',
      brief: 'for any item a Task lists, record it',
      task: 'refactory_packages.dataset_probe.tasks:CountingTask',
    })
    expect(subprocess.spawns[0]!.argv[subprocess.spawns[0]!.argv.indexOf('--task') + 1]).toBe(
      'refactory_packages.dataset_probe.tasks:CountingTask',
    )
  })
})

describe('running a mode once from the panel', () => {
  it('forwards a single run to the console\'s run verb, with the gate answers', async () => {
    // `start` is the optimizer; a person who has just created a mode needs to *use*
    // it, and the console had no verb for that at all.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, {
      action: 'run',
      mode: 'buildroot_pack/build',
      intent: 'discover the recipe',
      approve: ['approve=proceed'],
      headless: true,
      model: 'deepseek-v4-pro',
    })
    const argv = subprocess.spawns[0]!.argv
    expect(argv.slice(0, 4)).toEqual([
      '/repo/.venv/bin/python', '-m', 'refactory_dsh.console', 'run',
    ])
    expect(argv).toContain('buildroot_pack/build')
    expect(argv[argv.indexOf('--intent') + 1]).toBe('discover the recipe')
    expect(argv[argv.indexOf('--verdict') + 1]).toBe('approve=proceed')
    expect(argv[argv.indexOf('--model') + 1]).toBe('deepseek-v4-pro')
  })

  it('reports where a single run left its artifacts and what it recorded', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({
      ok: true,
      exit_code: 0,
      result: { output: '' },
      run: {
        run_id: 'buildroot_pack-build',
        state: 'finished',
        workdir: '/repo/.factory/interactive/buildroot_pack-build',
        facts: { name: 'buildroot_pack', closeness: 1.0 },
      },
    }))
    const rendered = text(await call(ctx, { action: 'run', mode: 'buildroot_pack/build' }))
    expect(rendered).toContain('Ran buildroot_pack/build')
    expect(rendered).toContain('/repo/.factory/interactive/buildroot_pack-build')
    expect(rendered).toContain('finished')
    expect(rendered).toContain('closeness: 1')
  })

  it('needs a mode to run', async () => {
    const { ctx } = await setup()
    const failed = await call(ctx, { action: 'run' })
    expect(failed.isError).toBe(true)
    expect(text(failed)).toContain('action "run" needs `mode`')
  })
})

describe('checking the input path and the artifact', () => {
  it('forwards the sample knob values the validation run should use', async () => {
    // The smoke runs the declared defaults, which for an input-taking mode is the
    // built-in fixture: the path the brief asked for goes untested without a sample.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, {
      action: 'create',
      name: 'p',
      brief: 'b',
      inputs: ['package'],
      sample: ['package=/tmp/demo_pkg'],
      approve: ['gate_strategy=proceed'],
      headless: true,
    })
    const argv = subprocess.spawns[0]!.argv
    expect(argv).toContain('--sample')
    expect(argv).toContain('package=/tmp/demo_pkg')
  })

  it('can continue a failed run instead of starting over', async () => {
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, { action: 'edit', mode: 'p/m', name: 'p', brief: 'b', resume: '/repo/.factory/edit/p' })
    const argv = subprocess.spawns[0]!.argv
    expect(argv[argv.indexOf('--resume') + 1]).toBe('/repo/.factory/edit/p')
  })
})

describe('seeding a search with the knobs a mode needs', () => {
  it('forwards knob values to start, one --knob per value', async () => {
    // A mode whose inputs are frozen knobs cannot be searched at all without them:
    // every candidate would run the declared defaults and score the same thing.
    const { ctx, subprocess } = await setup()
    subprocess.handler = () => runResult(JSON.stringify({ ok: true, exit_code: 0, result: {} }))
    await call(ctx, {
      action: 'start',
      mode: 'cve_patch_pack/patch',
      knobs: ['source_root=/tmp/demo_pkg', 'test_command=pytest -q'],
    })
    const argv = subprocess.spawns[0]!.argv
    expect(argv.filter(part => part === '--knob')).toHaveLength(2)
    expect(argv).toContain('source_root=/tmp/demo_pkg')
    expect(argv).toContain('test_command=pytest -q')
  })
})

it('retains complete mode constraints, role metadata and coverage warnings in model-visible output', async () => {
  const { ctx, subprocess } = await setup()
  const mode = { id: 'fixture/new', startable: true, parameters: {
    extra: { kind: 'int', default: 3, value: 3, range: [1, 8], values: [], frozen: false, description: 'Example count' },
  }, nodes: [{ id: 'emit', role: 'builder', kind: 'agent' }], coverage_warnings: ['Missing guidance'] }
  subprocess.handler = () => runResult(JSON.stringify([mode]))
  const rendered = text(await call(ctx, { action: 'modes' }))
  expect(rendered).toContain(JSON.stringify(mode))
})

it('read-only help/docs keep identity and bounded page text in the actual tool output', async () => {
  const { ctx, subprocess } = await setup()
  subprocess.handler = () => runResult(JSON.stringify({ build_id: 'running-build', documentation_id: 'running-docs', text: 'Worker routing', next_offset: 14 }))
  const answer = await call(ctx, { action: 'docs', page: 'refactory/docs/worker-routing.md', limit: 14 })
  expect(text(answer)).toContain('running-build')
  expect(text(answer)).toContain('next_offset')
  expect(subprocess.spawns[0]!.argv).toContain('--page')
  expect(subprocess.spawns[0]!.argv).not.toContain('bash')
})
