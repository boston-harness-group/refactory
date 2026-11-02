/**
 * TS tests for addItemRecord / item_records.json writing (F6 / Brief item 3b).
 *
 * Tests that:
 * - addItemRecord stores records retrievable by getItemRecords
 * - item_records.json is written with all records when the spine finishes
 */
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { afterEach, describe, expect, it } from 'vitest'
import { Spine } from '../src/spine.ts'
import * as spinePlugin from '../src/index.ts'
import type { Config } from '../src/index.ts'
import type { WorkflowGraph } from '../src/types.ts'

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-item-records-'))
  dirs.push(dir)
  return dir
}

function simpleDataGraph(): WorkflowGraph {
  return {
    name: 'records-test',
    start_node: 'data',
    nodes: {
      data: {
        _type: 'DataNode', id: 'data',
        task_ref: 'probe', subgraph_entry: 'work', subgraph_exit: 'work',
        reads: [], writes: ['current_item.json'],
      },
      work: { _type: 'FnNode', id: 'work', command: 'true', reads: [], writes: [] },
      after: { _type: 'FnNode', id: 'after', command: 'true', reads: [], writes: [] },
    },
    edges: [
      { source: 'data', target: 'after' },
    ],
  }
}

describe('addItemRecord / getItemRecords', () => {
  it('addItemRecord stores record by item id', () => {
    const spine = new Spine(simpleDataGraph(), { maxIterations: 2 })

    const record = {
      id: 'item-1',
      passed: true,
      score: 1.0,
      exchanges: [{ node: 'agent', prompt: 'hello', response: 'world' }],
      prediction: 'answer',
      label: 'correct',
      label_text: 'Label text',
      fail_reason: null,
      error_kind: null,
      error_text: null,
      metadata: { tag: 'test' },
    }

    spine.addItemRecord(record)

    const records = spine.getItemRecords()
    expect(records).toHaveLength(1)
    expect(records[0]).toMatchObject({
      id: 'item-1',
      passed: true,
      score: 1.0,
      prediction: 'answer',
      label: 'correct',
    })
    expect(records[0]!.exchanges).toEqual([{ node: 'agent', prompt: 'hello', response: 'world' }])
  })

  it('getItemRecords returns copies (not references to internal state)', () => {
    const spine = new Spine(simpleDataGraph(), { maxIterations: 2 })
    spine.addItemRecord({ id: 'a', passed: true })
    const first = spine.getItemRecords()
    spine.addItemRecord({ id: 'b', passed: false })
    const second = spine.getItemRecords()
    // The first snapshot should not have been mutated.
    expect(first).toHaveLength(1)
    expect(second).toHaveLength(2)
  })

  it('multiple records accumulate across data-node passes', () => {
    const spine = new Spine(simpleDataGraph(), { maxIterations: 2 })

    spine.addItemRecord({ id: 'alpha', passed: true, score: 1.0 })
    spine.addItemRecord({ id: 'beta', passed: false, score: 0.0, fail_reason: 'wrong' })
    spine.addItemRecord({ id: 'gamma', passed: true, score: 0.8 })

    const records = spine.getItemRecords()
    expect(records).toHaveLength(3)
    expect(records.map(r => r.id)).toEqual(['alpha', 'beta', 'gamma'])
  })
})

// ── item_records.json written to disk ──────────────────────────────────────

/** Loader shell that answers each data verb with the supplied item list. */
function loaderShell(items: Array<Record<string, unknown>>) {
  return {
    resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: 10_000,
      onExpiry: 'kill',
      stdoutMaxBytes: 1_000_000,
      env: request.env,
    }),
    execute: async (spec: { command: string }) => ({
      result: async () => {
        const verb = spec.command.split(' ')[3] ?? ''
        const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
        const item = items[index] ?? {}
        const answers: Record<string, unknown> = {
          list: { count: items.length, ids: items.map(i => i.id) },
          begin: item,
          end: {
            id: item.id,
            passed: true,
            score: 1.0,
            exchanges: [{ node: 'work', prompt: 'p', response: 'r' }],
            prediction: 'pred',
            label: 'lbl',
            label_text: 'Label',
            fail_reason: null,
            error_kind: null,
            error_text: null,
            metadata: { tag: 'test' },
          },
        }
        const text = verb in answers ? JSON.stringify(answers[verb]) : ''
        return {
          exitCode: 0, signal: null, timedOut: false, aborted: false, timeoutMs: 10_000,
          stdout: { text, truncated: false },
          stderr: { text: '', truncated: false },
        }
      },
    }),
  }
}

describe('item_records.json on disk', () => {
  it('item_records.json is written with all records when the spine finishes', async () => {
    const workdir = scratch()
    const items = [{ id: 'alpha' }, { id: 'beta' }]
    const config: Config = {
      graph: simpleDataGraph(),
      toolPrefix: 'wf',
      maxIterations: 20,
      interactiveOpen: false,
      openCommand: '',
      openArgs: [],
      executeCommands: true,
      commandEnv: {},
      commandTimeoutMs: 10_000,
      commandSandboxMode: 'danger-full-access',
      dataCommand: 'python -m refactory_dsh.items',
      steerCommand: '',
      steerArgs: [],
      workdir,
      ceoDeny: [],
    }
    const root = new Context()
    await root.plugin(SystemPrompt)
    await root.plugin(ToolRuntime)
    root.provide('shell', loaderShell(items))
    await root.plugin(spinePlugin, config)

    async function call(name: string, args: unknown = {}): Promise<unknown> {
      const def = root.tools.get(name)
      if (!def) throw new Error(`tool ${name} not registered`)
      return def.execute(args, {} as never)
    }

    // Drive the spine to completion: list → begin(0) → work → done → end(0) → begin(1) → work → done → end(1) → done
    let status = 'unknown'
    for (let step = 0; step < 100; step += 1) {
      const next = await call('wf_next') as { status?: string; nodes?: Array<{ id: string }> }
      status = next.status ?? 'unknown'
      if (status === 'done') break
      if (status !== 'ready') break
      for (const node of (next.nodes ?? [])) {
        await call('wf_done', { node: node.id })
      }
    }
    expect(status).toBe('done')

    // item_records.json must exist and contain both records.
    const recordsPath = join(workdir, 'item_records.json')
    expect(existsSync(recordsPath)).toBe(true)
    const records = JSON.parse(readFileSync(recordsPath, 'utf8'))
    expect(records).toHaveLength(2)
    expect(records.map((r: Record<string, unknown>) => r.id)).toEqual(['alpha', 'beta'])
    // Verify the shape: each record should have the fields command_end returns.
    for (const record of records) {
      expect(record).toHaveProperty('id')
      expect(record).toHaveProperty('passed')
      expect(record).toHaveProperty('score')
      expect(record).toHaveProperty('exchanges')
      expect(record).toHaveProperty('prediction')
      expect(record).toHaveProperty('label')
    }
  })
})

// ── Records from the fallback end paths reach item_records (F6 pins) ──────

type ShellAnswer = { exitCode?: number; timedOut?: boolean; stdout?: string; stderr?: string }

/**
 * A loader shell whose `end` answers are scripted per call, so a test can make an
 * end time out, exit non-zero, or a subgraph node fail, and check that the record
 * the fallback `end --failed` prints is still kept.
 */
function scriptedShell(
  items: Array<Record<string, unknown>>,
  script: (command: string, verb: string, index: number, calls: number) => ShellAnswer | undefined,
  log: string[],
) {
  let calls = 0
  return {
    resolve: (request: { command: string; workdir?: string; env?: Record<string, string> }) => ({
      command: request.command,
      workdir: request.workdir ?? process.cwd(),
      timeoutMs: 10_000,
      onExpiry: 'kill',
      stdoutMaxBytes: 1_000_000,
      env: request.env,
    }),
    execute: async (spec: { command: string }) => ({
      result: async () => {
        log.push(spec.command)
        calls += 1
        const verb = spec.command.split(' ')[3] ?? ''
        const index = Number(spec.command.match(/--index (\d+)/)?.[1] ?? 0)
        const item = items[index] ?? {}
        const failed = spec.command.includes('--failed')
        const answers: Record<string, unknown> = {
          list: { count: items.length, ids: items.map(i => i.id) },
          begin: item,
          end: { id: item.id, passed: !failed, score: failed ? 0 : 1, fail_reason: failed ? 'failed' : null },
        }
        const scripted = script(spec.command, verb, index, calls)
        const text = scripted?.stdout ?? (verb in answers ? JSON.stringify(answers[verb]) : '')
        return {
          exitCode: scripted?.exitCode ?? 0, signal: null, timedOut: scripted?.timedOut ?? false,
          aborted: false, timeoutMs: 10_000,
          stdout: { text, truncated: false },
          stderr: { text: scripted?.stderr ?? '', truncated: false },
        }
      },
    }),
  }
}

async function runToEnd(shell: object): Promise<string> {
  const workdir = scratch()
  const config: Config = {
    graph: simpleDataGraph(),
    toolPrefix: 'wf',
    maxIterations: 20,
    interactiveOpen: false,
    openCommand: '',
    openArgs: [],
    executeCommands: true,
    commandEnv: {},
    commandTimeoutMs: 10_000,
    commandSandboxMode: 'danger-full-access',
    dataCommand: 'python -m refactory_dsh.items',
    steerCommand: '',
    steerArgs: [],
    workdir,
    ceoDeny: [],
  }
  const root = new Context()
  await root.plugin(SystemPrompt)
  await root.plugin(ToolRuntime)
  root.provide('shell', shell)
  await root.plugin(spinePlugin, config)
  const def = root.tools.get('wf_next')
  if (!def) throw new Error('wf_next not registered')
  let status = 'unknown'
  for (let step = 0; step < 20 && status !== 'done'; step += 1) {
    const next = await def.execute({}, {} as never) as { status?: string }
    status = next.status ?? 'unknown'
    if (status !== 'ready' && status !== 'done') break
  }
  expect(status).toBe('done')
  return workdir
}

function recordsOn(workdir: string): Array<Record<string, unknown>> {
  return JSON.parse(readFileSync(join(workdir, 'item_records.json'), 'utf8'))
}

describe('fallback end records', () => {
  const items = [{ id: 'alpha' }, { id: 'beta' }]

  it('keeps the record when end times out and the fallback end --failed answers', async () => {
    const log: string[] = []
    const shell = scriptedShell(items, (command, verb, index) =>
      verb === 'end' && index === 0 && !command.includes('--failed') ? { timedOut: true } : undefined, log)
    const workdir = await runToEnd(shell)
    expect(log.filter(c => c.includes(' end ') && c.includes('--index 0'))).toHaveLength(2)
    const records = recordsOn(workdir)
    expect(records.map(r => r.id)).toEqual(['alpha', 'beta'])
    expect(records[0]).toMatchObject({ id: 'alpha', passed: false })
  })

  it('keeps the record when end exits non-zero (not all-fail) and end --failed answers', async () => {
    const log: string[] = []
    const shell = scriptedShell(items, (command, verb, index) =>
      verb === 'end' && index === 0 && !command.includes('--failed')
        ? { exitCode: 1, stdout: '', stderr: 'verify crashed' } : undefined, log)
    const workdir = await runToEnd(shell)
    const endsForAlpha = log.filter(c => c.includes(' end ') && c.includes('--index 0'))
    expect(endsForAlpha).toHaveLength(2)
    expect(endsForAlpha[1]).toContain('--failed')
    const records = recordsOn(workdir)
    expect(records.map(r => r.id)).toEqual(['alpha', 'beta'])
    expect(records[0]).toMatchObject({ id: 'alpha', passed: false })
  })

  it('keeps the record when a subgraph node fails and its end --failed times out once', async () => {
    const log: string[] = []
    let subgraphRuns = 0
    let failedEnds = 0
    const shell = scriptedShell(items, (command, verb, index) => {
      if (verb === '') {
        subgraphRuns += 1
        return subgraphRuns === 1 ? { exitCode: 1, stderr: 'subgraph node failed' } : undefined
      }
      if (verb === 'end' && index === 0 && command.includes('--failed')) {
        failedEnds += 1
        return failedEnds === 1 ? { timedOut: true } : undefined
      }
      return undefined
    }, log)
    const workdir = await runToEnd(shell)
    expect(failedEnds).toBe(2)
    const records = recordsOn(workdir)
    expect(records.map(r => r.id)).toEqual(['alpha', 'beta'])
    expect(records[0]).toMatchObject({ id: 'alpha', passed: false })
  })

  it('keeps the record when a subgraph node fails and its end --failed answers', async () => {
    let subgraphRuns = 0
    const shell = scriptedShell(items, (_command, verb) => {
      if (verb !== '') return undefined
      subgraphRuns += 1
      return subgraphRuns === 1 ? { exitCode: 1, stderr: 'subgraph node failed' } : undefined
    }, [])
    const records = recordsOn(await runToEnd(shell))
    expect(records.map(r => r.id)).toEqual(['alpha', 'beta'])
    expect(records[0]).toMatchObject({ id: 'alpha', passed: false })
  })

  it('writes the item records into the status file', async () => {
    const workdir = await runToEnd(scriptedShell(items, () => undefined, []))
    const status = JSON.parse(readFileSync(join(workdir, '.refactory/spine-status.json'), 'utf8'))
    expect(status.status).toBe('done')
    expect((status.item_records as Array<Record<string, unknown>>).map(r => r.id)).toEqual(['alpha', 'beta'])
  })
})
