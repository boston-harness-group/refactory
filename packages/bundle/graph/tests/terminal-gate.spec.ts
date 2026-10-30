/**
 * dm-255: a user gate put to the person at the terminal, in the composition `refactory ceo` boots.
 *
 * `refactory ceo` runs `dsh --profile graph`: dsh-base's rows, then this bundle's patch. The
 * spine used to read the user-questions service once, when it was applied, and the loader
 * applies rows as their modules arrive, so on a cold start it found none and refused every
 * user gate ("this composition mounts no user-questions channel") while the terminal answerer
 * sat mounted beside it. The CEO then invented a command to answer it with.
 *
 * Each case boots that composition in a process of its own (`terminal-gate-run.mjs`), as each
 * `refactory ceo` is, against the built packages (`pnpm run build`), and plays the CEO's one
 * call at a user gate — `wf_verdict` with the gate alone — against a stand-in terminal. The
 * fix is in the engine, so it holds for every person-answered gate: a generic one, and the
 * user gates of the shipped design, create, edit and review modes, as lowered.
 */
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'

const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url))
const snapshots = join(repoRoot, 'refactory', 'tests', 'lowering_snapshots')
const runner = fileURLToPath(new URL('./terminal-gate-run.mjs', import.meta.url))

type Graph = { name: string; start_node: string; nodes: Record<string, any>; edges: Array<Record<string, any>>; [key: string]: unknown }

interface Outcome {
  verdict?: Record<string, unknown>
  error?: string
  cursor?: string[]
  next?: Record<string, unknown>
  status?: { status: string; completed?: string[]; error?: string }
  shown: string
  workdir: string
}

const scratches: string[] = []

afterEach(() => {
  for (const dir of scratches.splice(0)) rmSync(dir, { recursive: true, force: true })
})

/** A lowered mode, cut to what its gate can reach, so the run starts at the gate. */
function fromGate(snapshot: string, gate: string): Graph {
  const graph = JSON.parse(readFileSync(join(snapshots, `${snapshot}.graph.json`), 'utf8')) as Graph
  // A fork names its branches, and a join its sources, on the node rather than on edges.
  const next = (id: string): string[] => [
    ...graph.edges.filter(edge => edge.source === id).map(edge => String(edge.target)),
    ...(graph.nodes[id]?.targets ?? []), ...(graph.nodes[id]?.sources ?? []),
  ]
  const kept = new Set([gate])
  for (let grew = true; grew;) {
    grew = false
    for (const id of [...kept]) {
      for (const target of next(id)) {
        if (!kept.has(target)) { kept.add(target); grew = true }
      }
    }
  }
  return {
    ...graph,
    start_node: gate,
    nodes: Object.fromEntries(Object.entries(graph.nodes).filter(([id]) => kept.has(id))),
    edges: graph.edges.filter(edge => kept.has(edge.source) && kept.has(edge.target)),
  }
}

/** A gate a person answers, and the one harmless step its `proceed` leads to. */
const GENERIC: Graph = {
  name: 'generic',
  start_node: 'approve',
  nodes: {
    approve: { _type: 'GateNode', id: 'approve', evaluator_type: 'user', reads: [], writes: [],
      gate_prompt: 'Publish the note?' },
    publish: { _type: 'FnNode', id: 'publish', command: 'true', reads: [], writes: [] },
  },
  edges: [{ source: 'approve', target: 'publish', condition: 'proceed' }],
}

/** Run one terminal in a fresh process: `chat` is `--chat`, and `answer` is typed at the gate. */
function terminal(graph: Graph, options: { chat: boolean; answer?: string; thenNext?: boolean }): Outcome {
  const scratch = mkdtempSync(join(tmpdir(), 'dsh-terminal-gate-'))
  scratches.push(scratch)
  const graphPath = join(scratch, 'graph.json')
  writeFileSync(graphPath, JSON.stringify(graph))
  const ran = spawnSync(process.execPath, [runner, scratch, graphPath, options.chat ? '1' : '0',
    options.answer ?? '', options.thenNext ? '1' : '0'], { encoding: 'utf8', timeout: 90_000 })
  const last = ran.stdout.trim().split('\n').pop() ?? ''
  try {
    return JSON.parse(last) as Outcome
  } catch {
    throw new Error(`the terminal run printed no outcome (exit ${ran.status}):\n${ran.stdout}\n${ran.stderr}`)
  }
}

describe('a user gate at the terminal (dm-255)', () => {
  it('puts the gate to the person, prints its card and options, and routes the answer', () => {
    const outcome = terminal(GENERIC, { chat: true, answer: '1', thenNext: true })
    expect(outcome.error).toBeUndefined()
    expect(outcome.verdict).toMatchObject({ status: 'routed', gate: 'approve', verdict: 'proceed', answered_by: 'user' })
    expect(outcome.shown).toContain('[gate approve] Publish the note?')
    expect(outcome.shown).toContain('1. proceed')
    expect(outcome.shown).toContain('2. halt')
    expect(outcome.next).toMatchObject({ status: 'done' })
    expect(outcome.status?.status).toBe('done')
    expect(outcome.status?.completed).toEqual(expect.arrayContaining(['approve', 'publish']))
  }, 120_000)

  it.each([
    ['default_pack-design', 'gate_strategy', 'Approve the design before the artifact is written?', 'pick_corrections'],
    ['default_pack-create', 'gate_strategy', 'Approve the strategy before any code is written?', 'design'],
    ['default_pack-edit', 'gate_strategy', 'Approve the strategy before any code is written?', 'design'],
    ['review_pack-review', 'approve', 'the person approves the draft', 'publish'],
  ])('puts %s\'s %s to the person and routes their proceed', (snapshot, gate, ask, next) => {
    const outcome = terminal(fromGate(snapshot, gate), { chat: true, answer: 'proceed' })
    expect(outcome.error).toBeUndefined()
    expect(outcome.verdict).toMatchObject({ status: 'routed', gate, verdict: 'proceed', answered_by: 'user' })
    expect(outcome.shown).toContain(`[gate ${gate}] ${ask}`)
    expect(outcome.shown).toMatch(/\n {2}\d+\. proceed\n/)
    expect(outcome.cursor).toContain(next)
  }, 120_000)

  it('still refuses a user gate headless, naming the commands that answer it', () => {
    const outcome = terminal(GENERIC, { chat: false })
    const refusal = new RegExp('gate approve is answered by a person and could not be asked: no user-questions '
      + 'answerer accepted the request; answer it with `refactory steer --run [^`]+ --verdict '
      + 'approve=<proceed\\|halt> \\[--note "…"\\]`, then `refactory graph-control resume --run [^`]+`')
    expect(outcome.verdict).toBeUndefined()
    expect(outcome.error).toMatch(refusal)
    expect(outcome.status?.status).toBe('refused')
    expect(outcome.status?.error).toMatch(refusal)
    expect(outcome.shown).not.toContain('[gate approve]')
  }, 120_000)
})
