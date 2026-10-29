/**
 * The interactive workflow spine: a deterministic cursor/gate/mutation state
 * machine over a workflow graph. It owns the position ("what is next"),
 * gate routing, iteration caps, fork/join barriers, director expansion, and
 * graph edits — so the driving agent never chooses the next step and cannot
 * drift. Pure logic, no DSH imports.
 * @module @deepseek-ai/dsh-workflow-spine/spine
 */

import { retryEdges, retryKey } from './loops.ts'
import type { DirectorExpansion, WorkflowEdge, WorkflowGraph, WorkflowNode } from './types.ts'

/** The model-facing view of one node returned by {@link Spine.next}. */
export interface NodeView {
  id: string
  _type: string
  role?: string
  resource_role?: 'work' | 'final_validation'
  evaluator_type?: string
  /** For an `fn` gate: the command whose last stdout line is the verdict, and any lines before it the reason. */
  evaluator_command?: string
  gate_prompt?: string
  /**
   * For a gate: the verdict labels it accepts, derived from its outgoing
   * conditional edges. A gate with a `proceed` label may also fall through its
   * single unconditional edge.
   */
  verdicts?: string[]
  /** For a gate: this gate's own reloop cap, overriding the runner's default. */
  max_iterations?: number
  reads?: string[]
  writes?: string[]
  /** The task handed to the driving agent, for `AgentNode` and `LLMNode`. */
  prompt_template?: string
  /** `AgentNode`: optional system-prompt text, distinct from the user prompt and notes. */
  system_prompt?: string
  /** For a one-shot `LLMNode`: the model and sampling it was declared with. */
  model?: string
  temperature?: number
  content?: string
  command?: string
  command_timeout_s?: number
  targets?: string[]
  optional_targets?: string[]
  sources?: string[]
  expand?: DirectorExpansion
  /** For a `DataNode`: its resolved data source and the subgraph it iterates. */
  task_ref?: string
  source_path?: string
  source_format?: string
  inline_items?: Array<Record<string, unknown>>
  subgraph_entry?: string
  subgraph_exit?: string
  limit?: number
  /** For a `SelectionNode`: the selection strategy. */
  strategy?: string
  unsupported?: boolean
}

/**
 * A data node's iteration, in progress: the data node, how many items it selected,
 * and which item's pass is the one running.
 */
export interface DataPass {
  /** The data node being iterated. */
  node: string
  /** How many items the selection holds; zero finishes the node without a pass. */
  count: number
  /** The index of the item whose pass is running, or just finished. */
  index: number
  /** The subgraph's first node, where each pass starts. */
  entry: string
  /** The subgraph's last node; completing it ends one pass. */
  exit: string
  /** True once the exit completed: the pass is finished and awaits its verdict. */
  finished: boolean
}

/** A gate stopped the run, rather than completing an item branch. */
export interface TerminalHalt {
  gate: string
  reason: 'verdict' | 'max_iterations' | 'cancelled' | 'incomplete'
  /** Sink nodes the run never reached, when reason is 'incomplete'. */
  unreached?: string[]
}

/**
 * The result of {@link Spine.next}: the nodes due now, or the end of the run with
 * the count of what completed.
 */
export type SpineNext =
  | { status: 'ready'; nodes: NodeView[]; paused?: string[]; feedback?: Record<string, string>; completed?: number; skipped?: string[]; pass?: DataPass }
  | { status: 'done'; nodes?: never; feedback?: Record<string, string>; completed?: number; skipped?: string[]; pass?: DataPass; halted?: TerminalHalt }

/** How {@link Spine.amendPrompt} changes a node's prompt. */
export type PromptAmendmentMode = 'set' | 'append'

const DEFAULT_MAX_ITERATIONS = 3

/** Node types the spine advances through; others are marked unsupported. */
const EXECUTED_TYPES = new Set([
  'AgentNode', 'LLMNode', 'FnNode', 'Study', 'GateNode', 'ForkNode', 'JoinNode',
  'DataNode', 'SelectionNode',
])

/** Verdict labels the spine treats as control flow rather than a forward branch. */
export const RELOOP = 'reloop'
export const HALT = 'halt'
export const PROCEED = 'proceed'

/** Normalize a verdict/condition label. Gates and evaluators name outcomes freely. */
function label(value: string): string {
  return value.trim().toLowerCase()
}

/** The node field a prompt lives in, or `undefined` for a node type that has none. */
function promptField(node: WorkflowNode): string | undefined {
  if (node._type === 'AgentNode') return 'prompt_template'
  // An LLM node's view reports `system_prompt ?? instance_prompt`, so the
  // amendment has to land in whichever of the two that view reads.
  if (node._type === 'LLMNode') return node.system_prompt !== undefined ? 'system_prompt' : 'instance_prompt'
  return undefined
}

/** One node's current prompt, read from the field its view reports. */
function promptText(node: WorkflowNode | undefined): string {
  if (node === undefined) return ''
  const field = promptField(node)
  if (field === undefined) return ''
  const value = node[field]
  return typeof value === 'string' ? value : ''
}

/** One structural defect found by {@link validateGraph}. */
export interface GraphIssue {
  /** The node the issue is attached to, when it is node-local. */
  node?: string
  message: string
}

/**
 * Check a graph against the invariants the spine's cursor relies on. Returns an
 * empty array for a walkable graph. These are structural properties the walk
 * cannot recover from at runtime, so a caller that accepts an external
 * `graph.json` should reject the graph before starting a run.
 * @param graph - the workflow graph, as produced by refactory's `Workflow.to_dict()`.
 * @returns every issue found, in node order.
 */
export function validateGraph(graph: WorkflowGraph): GraphIssue[] {
  const issues: GraphIssue[] = []
  const ids = new Set(Object.keys(graph.nodes))
  if (!ids.has(graph.start_node)) {
    issues.push({ message: `start_node '${graph.start_node}' is not a node` })
  }
  const outgoing = new Map<string, WorkflowEdge[]>()
  for (const edge of graph.edges) {
    if (!ids.has(edge.source)) issues.push({ message: `edge source '${edge.source}' is not a node` })
    if (!ids.has(edge.target)) issues.push({ message: `edge target '${edge.target}' is not a node` })
    const list = outgoing.get(edge.source)
    if (list === undefined) outgoing.set(edge.source, [edge])
    else list.push(edge)
  }

  for (const [id, node] of Object.entries(graph.nodes)) {
    if (node.id !== id) {
      issues.push({ node: id, message: `node declares id '${node.id}' but is keyed as '${id}'` })
    }
    if (!EXECUTED_TYPES.has(node._type)) {
      issues.push({ node: id, message: `node type '${node._type}' is not executable` })
    }
    const edges = outgoing.get(id) ?? []
    const unconditional = edges.filter(edge => edge.condition == null)
    const conditional = edges.filter(edge => edge.condition != null)

    if (node._type === 'GateNode') {
      if (edges.length === 0) issues.push({ node: id, message: 'gate has no outgoing edge' })
      const seen = new Set<string>()
      for (const edge of conditional) {
        const key = label(edge.condition as string)
        if (seen.has(key)) issues.push({ node: id, message: `two edges carry the verdict '${key}'` })
        seen.add(key)
      }
      if (unconditional.length > 1) {
        issues.push({ node: id, message: 'gate has more than one unconditional edge' })
      }
    } else if (unconditional.length > 1) {
      issues.push({ node: id, message: 'node has more than one unconditional edge' })
    }

    if (node._type === 'ForkNode') {
      for (const target of node.targets ?? []) {
        if (!ids.has(target)) issues.push({ node: id, message: `fork target '${target}' is not a node` })
      }
    }
    if (node._type === 'JoinNode') {
      for (const source of node.sources ?? []) {
        if (!ids.has(source)) issues.push({ node: id, message: `join source '${source}' is not a node` })
      }
    }
    if (node._type === 'DataNode') {
      issues.push(...dataSubgraphIssues(graph, id, node, ids, outgoing))
    }
  }

  for (const issue of unreachableNodes(graph, ids, outgoing)) {
    issues.push({ node: issue, message: 'unreachable from start_node' })
  }
  try {
    retryEdges(graph.nodes, graph.edges)
  } catch (error) {
    issues.push({ message: error instanceof Error ? error.message : String(error) })
  }
  // Cross-container reloop check: a gate inside a fork body may not reloop
  // to a target outside that fork's body.
  issues.push(...crossContainerReloopIssues(graph, outgoing))
  return issues
}

/**
 * Detect reloop edges whose target lies outside the gate's innermost
 * container (fork). Returns issues for each violation.
 */
function crossContainerReloopIssues(
  graph: WorkflowGraph,
  outgoing: Map<string, WorkflowEdge[]>,
): GraphIssue[] {
  const issues: GraphIssue[] = []
  let retrySet: Set<string>
  try {
    retrySet = retryEdges(graph.nodes, graph.edges)
  } catch {
    return issues // retryEdges already reported its error
  }
  // For each fork, compute its body
  for (const [forkId, forkNode] of Object.entries(graph.nodes)) {
    if (forkNode._type !== 'ForkNode') continue
    const joinEdge = (outgoing.get(forkId) ?? []).find(e => e.condition == null)
    if (joinEdge === undefined) continue
    const joinId = joinEdge.target
    const body = forwardWalk(forkId, joinId, { outgoing: id => outgoing.get(id) ?? [], node: id => graph.nodes[id] }, retrySet)
    body.delete(forkId)
    body.delete(joinId)
    // Check if any gate in this body has a retry edge to a target outside the body
    for (const nodeId of body) {
      if (graph.nodes[nodeId]?._type !== 'GateNode') continue
      for (const edge of outgoing.get(nodeId) ?? []) {
        if (edge.condition == null) continue
        if (!retrySet.has(retryKey(edge.source, edge.target, edge.condition))) continue
        if (!body.has(edge.target)) {
          issues.push({
            node: nodeId,
            message: `gate ${nodeId} is inside container ${forkId}, but reloop target ${edge.target} is outside it; place the reloop target within the same container, or move the gate to the top level`,
          })
        }
      }
    }
  }
  return issues
}

/**
 * Every node reachable forward from `start`, `start` included: edges (less the
 * retry back-edges in `skip`), a remembered director successor, fork targets,
 * extra continuations the caller supplies, and a data node's body walked up to
 * its exit. A walk stops at `stop`. The one forward walk both the spine and the
 * load-time validator use, so the validator's container bodies are the spine's.
 */
function forwardWalk(
  start: string,
  stop: string | undefined,
  look: {
    outgoing: (id: string) => WorkflowEdge[]
    node: (id: string) => WorkflowNode | undefined
    successor?: (id: string) => string | undefined
    continuations?: (id: string) => string[]
  },
  skip?: Set<string>,
): Set<string> {
  const found = new Set<string>()
  const visited = new Set<string>()
  const queue = [{ id: start, stop }]
  for (const next of queue) {
    const { id } = next
    const key = JSON.stringify(next)
    if (visited.has(key)) continue
    visited.add(key)
    found.add(id)
    if (id === next.stop) continue
    const targets = look.outgoing(id)
      .filter(edge => skip === undefined || !skip.has(retryKey(edge.source, edge.target, edge.condition)))
      .map(edge => edge.target)
    const successor = look.successor?.(id)
    if (successor !== undefined) targets.push(successor)
    const body = look.node(id)
    if (body?._type === 'ForkNode') targets.push(...(body.targets ?? []))
    if (look.continuations !== undefined) targets.push(...look.continuations(id))
    queue.push(...targets.map(target => ({ id: target, stop: next.stop })))
    if (body?._type === 'DataNode' && body.subgraph_entry !== undefined) {
      queue.push({ id: body.subgraph_entry, stop: body.subgraph_exit })
    }
  }
  return found
}

/** Nodes the walk can never reach, following edges plus fork fan-out. */
function unreachableNodes(
  graph: WorkflowGraph,
  ids: Set<string>,
  outgoing: Map<string, WorkflowEdge[]>,
): string[] {
  if (!ids.has(graph.start_node)) return []
  const seen = new Set<string>([graph.start_node])
  const queue = [graph.start_node]
  while (queue.length > 0) {
    const id = queue.shift() as string
    const next = (outgoing.get(id) ?? []).map(edge => edge.target)
    const node = graph.nodes[id]
    if (node?._type === 'ForkNode') next.push(...(node.targets ?? []))
    // A data node's subgraph is reached through the data node, not through an edge:
    // without this the whole subgraph reads as unreachable.
    if (node?._type === 'DataNode' && (node.subgraph_entry ?? '') !== '') next.push(node.subgraph_entry as string)
    for (const target of next) {
      if (ids.has(target) && !seen.has(target)) {
        seen.add(target)
        queue.push(target)
      }
    }
  }
  return [...ids].filter(id => !seen.has(id))
}

/**
 * What a data node's subgraph must be for the cursor to iterate it.
 *
 * The pass ends by completing `subgraph_exit`, and a gate is completed with
 * `verdict()` rather than `complete()`, so a gate exit is a run that cannot
 * finish. The entry must reach the exit, because the walk only ever moves forward
 * from the entry; and the data node must not also have an explicit edge into its
 * own subgraph, which would run the first node twice per item.
 * @param graph - the whole graph, for the exit's type.
 * @param id - the data node id.
 * @param node - the data node.
 * @param ids - every node id in the graph.
 * @param outgoing - edges by source node.
 * @returns every issue found, in the order the checks make them.
 */
function dataSubgraphIssues(
  graph: WorkflowGraph,
  id: string,
  node: WorkflowNode,
  ids: Set<string>,
  outgoing: Map<string, WorkflowEdge[]>,
): GraphIssue[] {
  const issues: GraphIssue[] = []
  const entry = node.subgraph_entry ?? ''
  const exit = node.subgraph_exit ?? ''
  if (entry === '' || !ids.has(entry)) {
    issues.push({ node: id, message: `data node subgraph_entry '${entry}' is not a node` })
  }
  if (exit === '' || !ids.has(exit)) {
    issues.push({ node: id, message: `data node subgraph_exit '${exit}' is not a node` })
  } else if (graph.nodes[exit]?._type === 'GateNode') {
    issues.push({
      node: id,
      message: `data node subgraph_exit '${exit}' is a gate; a pass ends by completing its exit`,
    })
  }
  if (entry !== '' && ids.has(entry) && exit !== '' && ids.has(exit) && !reaches(entry, exit, ids, outgoing, graph)) {
    issues.push({ node: id, message: `data node subgraph_entry '${entry}' cannot reach subgraph_exit '${exit}'` })
  }
  const intoSubgraph = (outgoing.get(id) ?? []).some(edge => edge.target === entry)
  if (intoSubgraph) {
    issues.push({
      node: id,
      message: `data node has an edge into its own subgraph ('${entry}'), which would run it twice per item`,
    })
  }
  return issues
}

/** Whether `from` reaches `to` along edges plus fork fan-out. */
function reaches(
  from: string,
  to: string,
  ids: Set<string>,
  outgoing: Map<string, WorkflowEdge[]>,
  graph: WorkflowGraph,
): boolean {
  const seen = new Set<string>()
  const queue = [from]
  while (queue.length > 0) {
    const id = queue.shift() as string
    if (id === to) return true
    if (seen.has(id)) continue
    seen.add(id)
    const next = (outgoing.get(id) ?? []).map(edge => edge.target)
    const node = graph.nodes[id]
    if (node?._type === 'ForkNode') next.push(...(node.targets ?? []))
    for (const target of next) if (ids.has(target) && !seen.has(target)) queue.push(target)
  }
  return false
}

/** The runtime's facts invalidation and the cursor's corresponding restart. */
export interface RedoPlan {
  requested: string
  start: string
  nodes: string[]
  resetPass: boolean
  restartData: string[]
}

export class Spine {
  private readonly sourceGraph: string
  private readonly nodes: Map<string, WorkflowNode>
  private edges: WorkflowEdge[]
  private readonly workflow: string
  private readonly projectDir: string
  private readonly maxIterations: number
  private cursor: string[]
  private afters: Map<string, string>
  private completed: string[]
  private skipped: string[]
  private skipProvenance: Map<string, { source: string; at: string }>
  private pendingSkipReceipts: Array<{ node: string; provenance: { source: string; at: string } }>
  private readonly gateVerdicts: Map<string, string>
  private readonly iterations: Record<string, number>
  private readonly feedback: Record<string, string>
  private readonly successors: Record<string, string>
  private readonly log: string[]
  private readonly promptEdits: Set<string>
  private readonly gateReasons: Record<string, { reason: string; at: number }>
  private gateReasonSeq: number
  private pass: DataPass | undefined
  private halted: TerminalHalt | undefined
  private readonly pauseBoundaries = new Set<string>()

  constructor(graph: WorkflowGraph, options?: { maxIterations?: number }) {
    this.sourceGraph = JSON.stringify(graph)
    this.nodes = new Map(Object.entries(graph.nodes))
    this.edges = JSON.parse(JSON.stringify(graph.edges)) as WorkflowEdge[]
    this.workflow = graph.name
    this.projectDir = graph.project_dir ?? ''
    this.maxIterations = options?.maxIterations ?? DEFAULT_MAX_ITERATIONS
    this.cursor = [graph.start_node]
    this.afters = new Map()
    this.completed = []
    this.skipped = []
    this.skipProvenance = new Map()
    this.pendingSkipReceipts = []
    this.gateVerdicts = new Map()
    this.iterations = {}
    this.feedback = {}
    this.successors = {}
    this.log = []
    this.promptEdits = new Set()
    this.gateReasons = {}
    this.gateReasonSeq = 0
    this.retryEdges()
    this.validateContainerReloops()
  }

  /** Recompute after expansion; remembered continuations are control edges too. */
  private retryEdges(): Set<string> {
    const continuations = Object.entries(this.successors).map(([source, target]) => ({ source, target }))
    return retryEdges(Object.fromEntries(this.nodes), [...this.edges, ...continuations])
  }

  private outgoing(nid: string): WorkflowEdge[] {
    return this.edges.filter(edge => edge.source === nid)
  }

  private unconditionalTarget(nid: string): string | undefined {
    return this.outgoing(nid).find(edge => edge.condition == null)?.target
  }

  private conditionalTarget(nid: string, condition: string): string | undefined {
    const wanted = label(condition)
    return this.outgoing(nid).find(
      edge => edge.condition != null && label(edge.condition) === wanted,
    )?.target
  }

  /** A gate's own reloop cap when it declares one, else the runner's default. */
  private reloopCap(gate: string): number {
    const declared = this.nodes.get(gate)?.max_iterations
    return typeof declared === 'number' ? declared : this.maxIterations
  }

  /**
   * The reloops a gate has spent of its cap, and the node a reloop goes back to; undefined
   * for a gate without a counted reloop edge. A reloop past the cap halts the run.
   * @param gate - the gate node id.
   * @returns what is spent and allowed, and the reloop's target.
   */
  reloops(gate: string): { used: number; cap: number; target: string } | undefined {
    const edge = this.outgoing(gate).find(candidate =>
      candidate.condition != null && label(candidate.condition) === RELOOP)
    if (edge === undefined || !this.retryEdges().has(retryKey(gate, edge.target, edge.condition))) return undefined
    return { used: this.iterations[gate] ?? 0, cap: this.reloopCap(gate), target: edge.target }
  }

  /**
   * Un-complete the nodes a retry from `gate` back to `target` runs again: exactly the
   * cycle, every node reachable from `target` that can also reach `gate`, along non-retry
   * edges, fork targets and join sources (`RunRecord._uncomplete_between` on the walker).
   * A target that is itself a gate (design mode's send-backs onto `pick_corrections`,
   * dm-226) reopens what each of its routes leads to inside the cycle, and nothing outside.
   */
  private uncompleteBetween(target: string, gate: string): void {
    const retries = this.retryEdges()
    const forward = new Map<string, string[]>()
    const backward = new Map<string, string[]>()
    const link = (from: string, to: string): void => {
      forward.set(from, [...(forward.get(from) ?? []), to])
      backward.set(to, [...(backward.get(to) ?? []), from])
    }
    for (const [id, node] of this.nodes) {
      for (const edge of this.outgoing(id)) {
        if (!retries.has(retryKey(id, edge.target, edge.condition))) link(id, edge.target)
      }
      if (node._type === 'ForkNode') for (const child of node.targets ?? []) link(id, child)
      if (node._type === 'JoinNode') for (const source of node.sources ?? []) link(source, id)
      if (node._type === 'DataNode' && node.subgraph_entry !== undefined) link(id, node.subgraph_entry)
    }
    const reach = (start: string, next: Map<string, string[]>): Set<string> => {
      const seen = new Set<string>([start])
      const queue = [start]
      for (const node of queue) {
        for (const other of next.get(node) ?? []) {
          if (!seen.has(other)) { seen.add(other); queue.push(other) }
        }
      }
      return seen
    }
    const after = reach(target, forward)
    const before = reach(gate, backward)
    const cycle = [...after].filter(node => node !== gate && (node === target || before.has(node)))
    for (const node of cycle) {
      this.completed = this.completed.filter(entry => entry !== node)
      this.gateVerdicts.delete(node)
    }
  }

  private describe(nid: string): NodeView {
    const node = this.nodes.get(nid)
    if (node === undefined) throw new Error(`unknown node ${nid}`)
    const view: NodeView = { id: nid, _type: node._type }
    if (node.role !== undefined) view.role = node.role
    if (node._type === 'GateNode') {
      if (node.evaluator_type !== undefined) view.evaluator_type = node.evaluator_type
      if (node.evaluator_command !== undefined) view.evaluator_command = node.evaluator_command
      if (node.command_timeout_s !== undefined) view.command_timeout_s = node.command_timeout_s
      view.gate_prompt = node.gate_prompt ?? ''
      view.reads = node.reads ?? []
      view.verdicts = this.acceptedVerdicts(nid)
      if (typeof node.max_iterations === 'number') view.max_iterations = node.max_iterations
    }
    if (node._type === 'AgentNode') {
      view.writes = node.writes ?? []
      view.reads = node.reads ?? []
      view.prompt_template = node.prompt_template ?? ''
      if (node.system_prompt !== undefined && node.system_prompt !== '') view.system_prompt = node.system_prompt
      if (node.expand !== undefined) view.expand = node.expand
      // Dispatched AgentNode: map dispatch_command to the view's command field
      // so commandOf() and runDeterministic() can see it.
      if (typeof node.dispatch_command === 'string' && node.dispatch_command !== '') {
        view.command = node.dispatch_command
        if (node.command_timeout_s !== undefined) view.command_timeout_s = node.command_timeout_s
      }
    }
    if (node._type === 'LLMNode') {
      // A one-shot model call is delivered the same way an agent node is: the
      // driving agent runs it and reports the node complete.
      view.writes = node.writes ?? []
      view.reads = node.reads ?? []
      view.prompt_template = node.system_prompt ?? node.instance_prompt ?? ''
      view.model = node.model ?? ''
      view.temperature = node.temperature ?? 0
    }
    if (node.resource_role !== undefined) view.resource_role = node.resource_role
    if (node._type === 'ForkNode') {
      view.targets = node.targets ?? []
      if (node.optional_targets) view.optional_targets = node.optional_targets
    }
    if (node._type === 'JoinNode') view.sources = node.sources ?? []
    if (node._type === 'FnNode' || node._type === 'Study') {
      view.writes = node.writes ?? []
      view.command = node.command ?? ''
      if (node.command_timeout_s !== undefined) view.command_timeout_s = node.command_timeout_s
      if (node.content !== undefined) view.content = node.content
    }
    if (node._type === 'DataNode') {
      // A DataNode loads data items and drives per-item execution of its
      // declared subgraph (subgraph_entry → subgraph_exit). The driving agent
      // materializes the items and iterates the subgraph once per item.
      view.writes = node.writes ?? []
      view.reads = node.reads ?? []
      view.task_ref = node.task_ref ?? ''
      view.source_path = node.source_path ?? ''
      view.source_format = node.source_format ?? ''
      view.inline_items = node.inline_items ?? []
      view.subgraph_entry = node.subgraph_entry ?? ''
      view.subgraph_exit = node.subgraph_exit ?? ''
      if (node.limit !== undefined) view.limit = node.limit
    }
    if (node._type === 'SelectionNode') {
      view.strategy = node.strategy ?? 'best_score'
      view.reads = node.reads ?? []
      view.writes = node.writes ?? []
    }
    if (!EXECUTED_TYPES.has(node._type)) view.unsupported = true
    return view
  }

  /**
   * One node's view, due or not.
   *
   * The plugin needs this to decide who answers a gate: an `fn` gate carries a
   * command and is run for the CEO, while a `user` gate names no command and must be
   * put to a person. Deciding that from the view is what keeps the runtime, rather
   * than the model's word, the thing that answers it.
   * @param nodeId - the node to describe.
   * @returns the node view.
   */
  view(nodeId: string): NodeView {
    return this.describe(nodeId)
  }

  /** Refuse a mutation unless this node is currently due, before any side effect. */
  assertDue(node: string): void {
    if (!this.nodes.has(node)) throw new Error(`unknown node ${node}`)
    const paused = this.pausedAt()
    if (paused.length > 0) throw new Error(`run paused before ${paused.join(', ')}; explicitly unpause the boundary after the person asks to continue`)
    if (this.halted !== undefined || !this.cursor.includes(node)) {
      throw new Error(`node ${node} is not due (current: ${this.cursor.join(', ') || 'none'}); use explicit redo to reopen completed work`)
    }
  }

  /**
   * The verdict labels a gate accepts, in edge order and lowercased.
   * @param gate - the gate node id.
   * @returns the labels its conditional edges carry; empty when it has none.
   */
  acceptedVerdicts(gate: string): string[] {
    return this.outgoing(gate)
      .filter(edge => edge.condition != null)
      .map(edge => label(edge.condition as string))
  }

  /** Follow explicit skip edits without completing work or discarding peers. */
  private advanceSkipped(skipped = this.skipped, dryRun = false): void {
    let cursor = [...this.cursor]
    const afters = new Map(this.afters)
    let pass = this.pass === undefined ? undefined : { ...this.pass }
    const seen = new Set<string>()
    while (!cursor.some(node => this.pauseBoundaries.has(node))) {
      const node = cursor.find(entry => skipped.includes(entry))
      if (node === undefined) break
      // Queue skip receipt for person-skipped nodes passed over by advanceSkipped
      if (!dryRun) {
        const prov = this.skipProvenance.get(node)
        if (prov?.source === 'person') {
          this.pendingSkipReceipts.push({ node, provenance: prov })
        }
      }
      const position = JSON.stringify([cursor, [...afters], pass?.finished])
      if (seen.has(position)) throw new Error('skipped nodes form a workflow cycle; unskip a step before continuing')
      seen.add(position)
      if (pass !== undefined && !pass.finished && node === pass.exit) {
        pass = { ...pass, finished: true }
        cursor = cursor.filter(entry => entry !== node)
      } else {
        const target = this.unconditionalTarget(node)
        cursor = [...new Set([...cursor.filter(entry => entry !== node), ...(target === undefined ? [] : [target])])]
        // Container drain check on local copies
        let drained = true
        while (drained) {
          drained = false
          for (const [container, continuation] of afters) {
            const body = this.containerBody(container, continuation)
            const hasInBody = cursor.some(pos => body.has(pos))
            const hasActivePass = pass !== undefined && body.has(pass.node)
            const hasPendingCont = [...afters].some(([c2, cont2]) => c2 !== container && body.has(cont2))
            if (!hasInBody && !hasActivePass && !hasPendingCont) {
              afters.delete(container)
              if (!cursor.includes(continuation)) cursor.push(continuation)
              drained = true
              break
            }
          }
        }
      }
    }
    if (!dryRun) {
      this.cursor = cursor
      this.afters = afters
      this.pass = pass
    }
  }

  /**
   * Graph sinks: nodes the scheduler has nowhere to go after. A node with an
   * outgoing edge continues along it; a node inside a fork's or an expanded
   * director's body continues at that container's continuation
   * ({@link continuationOf}); a data node's body exit returns to its data node.
   */
  private sinks(): string[] {
    const dataBodyExits = new Set<string>()
    for (const node of this.nodes.values()) {
      if (node._type === 'DataNode' && node.subgraph_exit !== undefined) {
        dataBodyExits.add(node.subgraph_exit)
      }
    }
    const result: string[] = []
    for (const id of this.nodes.keys()) {
      if (this.outgoing(id).length > 0) continue
      if (this.continuationOf(id) !== undefined) continue
      if (dataBodyExits.has(id)) continue
      result.push(id)
    }
    return result
  }

  /** The node(s) due now, or `done`. */
  next(): SpineNext {
    this.retryEdges()
    // Person-skip routing rule: route person-skipped conditional-only gates via proceed.
    for (const nid of [...this.cursor]) {
      if (!this.skipped.includes(nid)) continue
      const prov = this.skipProvenance.get(nid)
      if (prov?.source !== 'person') continue
      const body = this.nodes.get(nid)
      if (body?._type !== 'GateNode') continue
      if (this.unconditionalTarget(nid) !== undefined) continue
      if (this.outgoing(nid).length === 0) continue
      this.verdict(nid, 'proceed', 'skipped by the person')
      this.pendingSkipReceipts.push({ node: nid, provenance: prov })
    }
    this.advanceSkipped()
    const pass = this.pass === undefined ? {} : { pass: { ...this.pass } }
    if (this.halted !== undefined || (this.cursor.length === 0 && this.pass === undefined)) {
      if (this.halted === undefined) {
        // Cursor drained without an explicit halt. Check whether any graph sink
        // was completed or skipped; with several sinks (alternative ends),
        // reaching any one is the end.
        const graphSinks = this.sinks()
        const reached = (id: string): boolean => this.completed.includes(id) || this.skipped.includes(id)
        if (graphSinks.length > 0 && !graphSinks.some(reached)) {
          this.halted = { gate: '', reason: 'incomplete', unreached: graphSinks }
        }
      }
      return {
        status: 'done', completed: this.completed.length, skipped: [...this.skipped], ...pass,
        ...(this.halted === undefined ? {} : { halted: { ...this.halted } }),
      }
    }
    return {
      status: 'ready',
      ...(this.isPaused() ? { paused: this.pausedAt() } : {}),
      nodes: this.cursor.map(nid => this.describe(nid)),
      feedback: { ...this.feedback },
      ...pass,
    }
  }

  /** Arm or release one explicit stop boundary; unrelated stops remain intact. */
  pauseBefore(node: string, paused: boolean): void {
    if (!this.nodes.has(node)) throw new Error(`unknown node ${node}`)
    if (paused) this.pauseBoundaries.add(node)
    else this.pauseBoundaries.delete(node)
    this.log.push(`${paused ? 'pause before' : 'unpaused'} ${node}`)
  }

  /** The armed boundaries currently preventing execution. */
  pausedAt(): string[] {
    if (this.halted !== undefined) return []
    // A completed item's body has an empty cursor until its Task is graded.
    // Its container remains a real command boundary that can be paused.
    const boundary = this.cursor.length === 0 && this.pass !== undefined ? [this.pass.node] : this.cursor
    return boundary.filter(node => this.pauseBoundaries.has(node))
  }

  /** Whether the current cursor must yield to the person before any work runs. */
  isPaused(): boolean {
    return this.pausedAt().length > 0
  }

  /** Whether a gate stopped the run before its remaining work completed. */
  isHalted(): boolean {
    return this.halted !== undefined
  }

  /**
   * The pass a data node is in the middle of, or `undefined`.
   * @param node - a data node id; absent answers for whichever pass is running.
   * @returns the pass, if one is running.
   */
  passOf(node?: string): DataPass | undefined {
    if (this.pass === undefined) return undefined
    if (node !== undefined && this.pass.node !== node) return undefined
    return { ...this.pass }
  }

  /**
   * Start iterating a data node: the cursor moves into the subgraph's entry.
   *
   * The items themselves are the caller's: loading them is I/O, and which command
   * loads them is deployment state, not graph structure. A selection of zero items
   * is not a pass — the caller completes the node instead.
   * @param node - the data node id.
   * @param count - how many items the selection holds; must be positive.
   * @returns the spine's next state, whose cursor is the subgraph's entry.
   */
  startPass(node: string, count: number): SpineNext {
    const body = this.nodes.get(node)
    if (body === undefined) throw new Error(`unknown node ${node}`)
    if (body._type !== 'DataNode') throw new Error(`${node} is not a data node`)
    if (this.pass !== undefined) throw new Error(`${node} cannot start: ${this.pass.node} is iterating`)
    if (!Number.isInteger(count) || count <= 0) {
      throw new Error(`${node} was given ${String(count)} items; an empty selection completes the node`)
    }
    const entry = body.subgraph_entry ?? ''
    const exit = body.subgraph_exit ?? ''
    if (entry === '' || exit === '') throw new Error(`${node} declares no subgraph to iterate`)
    this.invalidateNodes(this.descendants(entry, exit), false)
    this.pass = { node, count, index: 0, entry, exit, finished: false }
    this.splicePosition(node, [entry])
    this.log.push(`data ${node}: ${count} item(s), first pass at ${entry}`)
    return this.next()
  }

  /**
   * End one item's pass: move to the next item, or complete the data node.
   *
   * Called after the pass's own verdict is in hand, so a pass that was never judged
   * cannot be recorded as done. The data node's successor is reached here, on the
   * last item, rather than by following the exit's out-edge.
   * @returns the spine's next state.
   */
  endPass(): SpineNext {
    const pass = this.pass
    if (pass === undefined) throw new Error('no data pass is running')
    if (this.halted !== undefined) throw new Error(`run halted at ${this.halted.gate}; its item did not finish`)
    // A terminal branch (for example an explicit refusal) also ends an item.
    // The runner must still obtain its Task verdict before advancing the pass.
    if (!pass.finished) {
      // Check if any pass body position remains in cursor
      const passBody = this.descendants(pass.entry, pass.exit)
      const hasPassPositions = this.cursor.some(pos => passBody.has(pos))
      if (hasPassPositions) {
        throw new Error(`item ${pass.index} of ${pass.node} has not finished`)
      }
    }
    if (pass.index + 1 < pass.count) {
      const index = pass.index + 1
      this.invalidateNodes(this.descendants(pass.entry, pass.exit), false)
      this.pass = { ...pass, index, finished: false }
      if (!this.cursor.includes(pass.entry)) this.cursor.push(pass.entry)
      this.log.push(`data ${pass.node}: item ${index + 1}/${pass.count}`)
      return this.next()
    }
    this.pass = undefined
    this.log.push(`data ${pass.node}: ${pass.count} item(s) done`)
    // Restore the enclosing data continuation before completing it. Its body
    // exhausted the cursor, but only this bookkeeping may resume the parent.
    if (!this.cursor.includes(pass.node)) this.cursor.push(pass.node)
    // This node entered before a boundary may have been armed. Closing that
    // admitted pass is bookkeeping, not a new entry across the pause boundary.
    return this.completeDue(pass.node)
  }

  /** Mark a non-gate node complete and advance. */
  complete(node: string): SpineNext {
    this.assertDue(node)
    return this.completeDue(node)
  }

  /** Completion after public admission, or the runner closing its admitted pass. */
  private completeDue(node: string): SpineNext {
    const body = this.nodes.get(node)
    if (body === undefined) throw new Error(`unknown node ${node}`)
    if (body._type === 'GateNode') throw new Error(`${node} is a gate — use verdict()`)
    if (this.skipped.includes(node)) return this.next()
    if (!this.completed.includes(node)) {
      this.completed.push(node)
      this.log.push(`completed ${node}`)
    }
    // Feedback is what a reloop asked of an agent's next run. That run has now
    // happened, so later views stop repeating it — a gate's reason can be a whole
    // check log, and it would otherwise ride along in every `next()` to the end.
    // A command or a fork never reads it, and runs inside the same drain, so its
    // feedback stays for the driver to see until the next reloop replaces it.
    if (body._type === 'AgentNode' || body._type === 'LLMNode') delete this.feedback[node]

    if (body._type === 'ForkNode') {
      const join = this.unconditionalTarget(node)
      if (join !== undefined) this.afters.set(node, join)
      this.splicePosition(node, [...(body.targets ?? [])])
      this.checkContainerDrained()
      return this.next()
    }

    // Completing the subgraph's exit ends one pass. The exit's own out-edge is not
    // followed: the data node's successor is reached when the last item is done, so
    // a pass cannot walk out of its subgraph into the nodes after the data node.
    if (this.pass !== undefined && !this.pass.finished && node === this.pass.exit) {
      this.pass = { ...this.pass, finished: true }
      this.cursor = this.cursor.filter(entry => entry !== node)
      this.checkContainerDrained()
      return this.next()
    }

    this.advance(node)
    this.checkContainerDrained()
    return this.next()
  }

  /**
   * Replace `node` with its successor, leaving any concurrent position alone.
   * A fork's branches each advance on their own; the join is reached only when
   * every branch has drained.
   */
  private advance(node: string): void {
    const target = this.unconditionalTarget(node)
    const rest = this.cursor.filter(entry => entry !== node)
    const replacement = target === undefined ? [] : [target]
    this.cursor = [...new Set([...rest, ...replacement])]
  }

  /**
   * Route a gate. `verdict` is `proceed`, `reloop`, `halt`, or any forward
   * outcome the gate names on its outgoing edges — a gate is a switch, not only
   * a binary decision. A `proceed` gate without a proceed edge falls through its
   * unconditional edge.
   * @param gate - the gate node id.
   * @param verdict - the outcome label; matched case-insensitively.
   * @param feedback - feedback attached to the reloop target (reloop only).
   * @returns the spine's next state.
   */
  verdict(gate: string, verdict: string, feedback = ''): SpineNext {
    if (this.nodes.get(gate)?._type !== 'GateNode') throw new Error(`${gate} is not a gate`)
    this.assertDue(gate)
    const outcome = label(verdict)
    if (outcome === '') throw new Error(`empty verdict for gate ${gate}`)
    // Validate the route before committing history: a rejected outcome must
    // leave the same cursor, verdicts, iterations and completion record.
    const record = (): void => {
      if (!this.completed.includes(gate)) this.completed.push(gate)
      this.gateVerdicts.set(gate, outcome)
      // A retry's feedback for this gate (a send-back onto `pick_corrections`, dm-226) was
      // for this visit; the gate has now decided, so later views stop repeating it.
      delete this.feedback[gate]
    }

    const selected = this.outgoing(gate).find(edge =>
      edge.condition != null && label(edge.condition) === outcome)
      ?? (outcome === PROCEED ? this.outgoing(gate).find(edge => edge.condition == null) : undefined)
    if (outcome === RELOOP && selected === undefined) throw new Error(`${gate} has no reloop edge`)
    const retry = selected !== undefined
      && this.retryEdges().has(retryKey(gate, selected.target, selected.condition))
    if (retry) {
      const target = selected!.target
      record()
      const cap = this.reloopCap(gate)
      const count = (this.iterations[gate] ?? 0) + 1
      this.iterations[gate] = count
      this.gateReasonSeq += 1
      this.gateReasons[gate] = { reason: feedback, at: this.gateReasonSeq }
      if (count > cap) {
        this.gateVerdicts.set(gate, HALT)
        this.halted = { gate, reason: 'max_iterations' }
        this.cursor = []
        this.afters = new Map()
        this.log.push(`gate ${gate}: max iterations -> halt`)
        return this.next()
      }
      // Refuse cross-container reloop (runtime backstop; also checked at load)
      const gateContainer = this.continuationOf(gate)
      if (gateContainer !== undefined) {
        const body = this.containerBody(gateContainer.container, gateContainer.continuation)
        if (!body.has(target)) {
          throw new Error(
            `gate ${gate} is inside container ${gateContainer.container}, but reloop target ${target} is outside it; `
            + `place the reloop target within the same container, or move the gate to the top level`
          )
        }
      }
      this.halted = undefined
      this.feedback[target] = feedback
      this.uncompleteBetween(target, gate)
      this.splicePosition(gate, [target])
      this.log.push(`gate ${gate}: ${outcome} -> ${target} (${count})`)
    } else if (outcome === HALT) {
      const target = this.conditionalTarget(gate, HALT)
      record()
      delete this.gateReasons[gate]
      if (target === undefined) {
        // Terminal halt: end the whole run
        this.halted = { gate, reason: 'verdict' }
        this.cursor = []
        this.afters = new Map()
      } else {
        // Forward halt edge: preserve siblings
        this.halted = undefined
        this.splicePosition(gate, [target])
        this.checkContainerDrained()
      }
      this.log.push(`gate ${gate}: halt`)
    } else {
      const target = this.conditionalTarget(gate, outcome)
        ?? (outcome === PROCEED ? this.unconditionalTarget(gate) : undefined)
      if (target === undefined) {
        const allowed = this.acceptedVerdicts(gate)
        throw new Error(`${gate} has no '${outcome}' edge (accepts ${allowed.join(', ')})`)
      }
      record()
      delete this.gateReasons[gate]
      this.halted = undefined
      this.splicePosition(gate, [target])
      this.checkContainerDrained()
      this.log.push(`gate ${gate}: ${outcome} -> ${target}`)
    }
    return this.next()
  }

  /** Stop at a due gate without certifying an evaluator result or taking an edge. */
  cancel(gate: string): SpineNext {
    this.assertDue(gate)
    if (this.nodes.get(gate)?._type !== 'GateNode') throw new Error(`${gate} is not a gate`)
    if (this.conditionalTarget(gate, HALT) !== undefined) {
      throw new Error(`${gate} has a forward halt edge; cancellation cannot choose its result`)
    }
    this.halted = { gate, reason: 'cancelled' }
    this.cursor = []
    this.afters = new Map()
    this.log.push(`gate ${gate}: cancelled`)
    return this.next()
  }

  /** Hot-plug a director's N child agents (a graph edit). */
  expand(director: string, entries: Array<Record<string, unknown>>): { expanded: string; children: string[]; next: SpineNext } {
    if (this.isPaused()) throw new Error(`run paused before ${this.pausedAt().join(', ')}; unpause before dispatching an expansion`)
    const node = this.nodes.get(director)
    const expansion = node?.expand
    if (expansion === undefined) throw new Error(`${director} has no expand annotation`)
    const slugField = expansion.slug_field
    const promptField = expansion.prompt_field
    const writesTemplate = expansion.writes_template

    // The children of a previous expansion carry edges from the director, so the
    // successor is the director's unconditional target that is **not** one of them;
    // otherwise a re-expansion would mistake its own child for the next node.
    const prefix = `${director}__`
    let successor = this.successors[director]
    if (successor === undefined) {
      const next = this.edges.find(edge =>
        edge.source === director && edge.condition == null && !edge.target.startsWith(prefix))?.target
      if (next !== undefined) {
        successor = next
        this.successors[director] = next
      }
    }

    // Revert the director's previous expansion: drop old children and their edges.
    // Joins that wait on this director, or on the children of a previous expansion,
    // are recorded with the sources they named before the revert erases them: a
    // re-expansion has to re-point them at the new children, and after the revert
    // nothing still says they were waiting on this director.
    const waitingJoins: Array<{ body: WorkflowNode; sources: string[] }> = []
    for (const body of this.nodes.values()) {
      const sources = body._type === 'JoinNode' ? body.sources : undefined
      if (sources === undefined) continue
      if (sources.some(source => source === director || source.startsWith(prefix))) {
        waitingJoins.push({ body, sources })
      }
    }
    for (const child of [...this.nodes.keys()]) {
      if (child.startsWith(prefix)) {
        this.nodes.delete(child)
        this.completed = this.completed.filter(entry => entry !== child)
      }
    }
    // Remove old children from cursor so re-expansion doesn't leave stale entries
    this.cursor = this.cursor.filter(entry => !entry.startsWith(prefix))
    this.edges = this.edges.filter(edge =>
      !(edge.source === director && edge.condition == null)
      && !(edge.source.startsWith(prefix) || edge.target.startsWith(prefix)),
    )
    for (const { body, sources } of waitingJoins) {
      body.sources = sources.filter(source => !source.startsWith(prefix))
    }

    const children: string[] = []
    entries.forEach((entry, index) => {
      const rawSlug = typeof entry[slugField] === 'string' ? entry[slugField] : `child${index}`
      const slug = rawSlug.trim().replace(/\//g, '-').replace(/ /g, '-')
      const prompt = typeof entry[promptField] === 'string' ? entry[promptField] : ''
      const writes = writesTemplate === '' ? [] : [writesTemplate.replace('{slug}', slug)]
      const childId = `${director}__${slug}`
      this.nodes.set(childId, { _type: 'AgentNode', id: childId, role: expansion.role, prompt_template: prompt, writes, reads: [] })
      children.push(childId)
    })

    for (const child of children) {
      this.edges.push({ source: director, target: child })
      // A successor is reached through the `afters` barrier, so the children
      // carry no edge to it: with several children, one edge would schedule the
      // successor as soon as the first child finished.
    }
    if (successor === undefined) {
      for (const { body, sources } of waitingJoins) {
        body.sources = sources
          .filter(source => source !== director && !source.startsWith(prefix))
          .concat(children)
      }
    }

    if (!this.completed.includes(director)) this.completed.push(director)
    if (successor !== undefined) this.afters.set(director, successor)
    this.splicePosition(director, children)
    this.log.push(`expanded ${director} -> ${children.join(',')}`)
    return { expanded: director, children, next: this.next() }
  }

  /** Admit a skip before the runtime retires its pending delivery attempts. */
  planSkip(node: string, opts?: { source?: string }): string[] {
    if (this.isPaused()) throw new Error(`run paused before ${this.pausedAt().join(', ')}; explicitly unpause before skipping work`)
    const body = this.nodes.get(node)
    if (body === undefined) throw new Error(`unknown node ${node}`)
    if (this.completed.includes(node)) throw new Error(`${node} already entered or completed; skip a due/future step, or explicitly redo completed work`)
    if (this.pass?.node === node) throw new Error(`${node} already has an active data pass; finish or skip its current body before changing the dataset`)
    if (body._type === 'GateNode' && this.unconditionalTarget(node) === undefined && this.outgoing(node).length > 0) {
      if (opts?.source !== 'person') {
        throw new Error(`${node} has conditional successors; route its verdict explicitly instead of an ambiguous skip`)
      }
    }
    // A skipped container bypasses its body, but still reaches its continuation.
    // Keep child skip flags independent so unskip(container) restores that body.
    const continuation = this.unconditionalTarget(node)
    const removed = body._type === 'ForkNode' || body._type === 'DataNode'
      ? [...this.descendants(node, continuation)].filter(id => id !== continuation)
      : [node]
    const stops = removed.filter(id => this.pauseBoundaries.has(id))
    if (stops.length > 0) throw new Error(`pause armed before ${stops.join(', ')}; explicitly unpause before skipping that boundary`)
    this.advanceSkipped([...this.skipped, node], true)
    return removed
  }

  /** Graph mutation: skip a node's work, retaining its unambiguous continuation. */
  skip(node: string, opts?: { source?: string; at?: string }): SpineNext {
    this.planSkip(node, opts)
    const skipped = [...this.skipped]
    const logLength = this.log.length
    const hadProvenance = this.skipProvenance.has(node)
    if (!this.skipped.includes(node)) {
      this.skipped.push(node)
      this.log.push(`skipped ${node}`)
    }
    const provenance = { source: opts?.source ?? 'ceo', at: opts?.at ?? new Date().toISOString() }
    this.skipProvenance.set(node, provenance)
    try {
      return this.next()
    } catch (error) {
      this.skipped = skipped
      this.log.length = logLength
      if (!hadProvenance) this.skipProvenance.delete(node)
      throw error
    }
  }

  /** Undo a skip. */
  unskip(node: string): SpineNext {
    if (this.skipped.includes(node)) {
      this.skipped = this.skipped.filter(entry => entry !== node)
      this.skipProvenance.delete(node)
      this.log.push(`unskipped ${node}`)
    }
    return this.next()
  }

  /**
   * The body of a container (fork or expanded director): the nodes between
   * `container` and `continuation`, walked without retry edges, minus the two
   * ends. One definition used by continuationOf, checkContainerDrained,
   * advanceSkipped, verdict cross-container check, and endPass scoping.
   */
  private containerBody(container: string, continuation: string): Set<string> {
    const body = this.descendants(container, continuation, { skipRetryEdges: true })
    body.delete(container)
    body.delete(continuation)
    return body
  }

  /**
   * Find the innermost container (fork or expanded director) whose body holds
   * `nodeId`, and the continuation that container pushes onto `afters`.
   *
   * These are the facts the scheduler itself uses: a ForkNode's continuation is
   * its unconditional edge target (what `completeDue` pushes), an expanded
   * director's is `this.successors[director]` (what `expand` pushes). A join's
   * `sources` are not consulted. The body is what `planSkip` removes for a fork,
   * `descendants(container, continuation)` minus the two ends, walked without
   * retry edges so a reloop inside a branch cannot pull in nodes before the
   * container. Bodies nest, so the innermost container has the smallest body.
   */
  private continuationOf(nodeId: string): { container: string; continuation: string } | undefined {
    const containers: Array<[string, string | undefined]> = [
      ...[...this.nodes].filter(([, body]) => body._type === 'ForkNode')
        .map(([id]): [string, string | undefined] => [id, this.unconditionalTarget(id)]),
      ...Object.entries(this.successors),
    ]
    let best: { container: string; continuation: string; size: number } | undefined
    for (const [container, continuation] of containers) {
      if (continuation === undefined) continue
      const body = this.containerBody(container, continuation)
      if (body.has(nodeId) && (best === undefined || body.size < best.size)) {
        best = { container, continuation, size: body.size }
      }
    }
    return best === undefined ? undefined : { container: best.container, continuation: best.continuation }
  }

  /**
   * Replace one position in the cursor with its successors, leaving all other
   * positions intact.
   */
  private splicePosition(old: string, replacements: string[]): void {
    const rest = this.cursor.filter(entry => entry !== old)
    this.cursor = [...new Set([...rest, ...replacements])]
  }

  /**
   * After any cursor mutation, check whether any container registered in
   * `afters` has no remaining positions inside its body. If so, move the
   * continuation from `afters` into `cursor`.
   *
   * A container is not drained while:
   * - any cursor position is inside its body (at any depth), OR
   * - any other registered container's continuation lies inside its body
   *   (so an inner join fires before the outer join).
   */
  private checkContainerDrained(): void {
    for (const [container, continuation] of this.afters) {
      const body = this.containerBody(container, continuation)
      const hasCursorInBody = this.cursor.some(pos => body.has(pos))
      // An active data pass keeps its data node's position virtually alive
      const hasActivePass = this.pass !== undefined && body.has(this.pass.node)
      // Also check whether a pending inner continuation lies in this body
      const hasPendingCont = [...this.afters].some(
        ([c2, cont2]) => c2 !== container && body.has(cont2),
      )
      if (!hasCursorInBody && !hasActivePass && !hasPendingCont) {
        this.afters.delete(container)
        if (!this.cursor.includes(continuation)) {
          this.cursor.push(continuation)
        }
        // After adding a continuation, re-check: that continuation might itself
        // be inside another container that is now drained.
        return this.checkContainerDrained()
      }
    }
  }

  /**
   * Validate at load time that no reloop edge crosses a container boundary.
   * A gate inside a fork branch (or expanded director body) may only reloop
   * to a target within the same container. This catches graph authoring errors
   * before a run starts.
   */
  private validateContainerReloops(): void {
    const retrySet = this.retryEdges()
    for (const edge of this.edges) {
      if (edge.condition == null) continue
      if (!retrySet.has(retryKey(edge.source, edge.target, edge.condition))) continue
      const gate = edge.source
      const target = edge.target
      const gateContainer = this.continuationOf(gate)
      if (gateContainer === undefined) continue
      const body = this.containerBody(gateContainer.container, gateContainer.continuation)
      if (!body.has(target)) {
        throw new Error(
          `gate ${gate} is inside container ${gateContainer.container}, but reloop target ${target} is outside it; `
          + `place the reloop target within the same container, or move the gate to the top level`
        )
      }
    }
  }

  /** All branches reachable from a node, including data bodies and fork targets. */
  private descendants(start: string, stop?: string, opts?: { skipRetryEdges?: boolean; followContinuations?: boolean }): Set<string> {
    const look: Parameters<typeof forwardWalk>[2] = {
      outgoing: id => this.outgoing(id),
      node: id => this.nodes.get(id),
      successor: id => this.successors[id],
    }
    if (opts?.followContinuations === true) {
      look.continuations = id => { const cont = this.continuationOf(id); return cont === undefined ? [] : [cont.continuation] }
    }
    return forwardWalk(start, stop, look, opts?.skipRetryEdges === true ? this.retryEdges() : undefined)
  }

  /** Clear current completion and gate decisions without erasing graph edits. */
  private invalidateNodes(affected: Set<string>, clearSkips = true, preserveIterations = false): void {
    this.completed = this.completed.filter(entry => !affected.has(entry))
    if (clearSkips) {
      this.skipped = this.skipped.filter(entry => !affected.has(entry))
      for (const id of affected) this.skipProvenance.delete(id)
    }
    for (const id of affected) {
      this.gateVerdicts.delete(id)
      if (!preserveIterations) delete this.iterations[id]
      delete this.feedback[id]
    }
  }

  /** Whether concurrent work (a fork's parallel branches) is in progress. */
  hasConcurrentWork(): boolean {
    return this.cursor.length > 1 || this.afters.size > 0
  }

  /** Set per-visit feedback for a node, visible in next() and cleared on complete. */
  setFeedback(node: string, text: string): void {
    if (!this.nodes.has(node)) throw new Error(`unknown node ${node}`)
    this.feedback[node] = text
    this.log.push(`feedback ${node}`)
  }

  /**
   * Describe invalidation before the runtime archives and clears current facts.
   *
   * A paused run refuses unless `whilePaused` is set. Only a person's rerun (a
   * graph-control applied at a boundary) and the checkpoint's descriptive plans set
   * it: the pause is the person's own, so they may reopen work behind it, while the
   * CEO's `wf_redo` must not move the cursor off a boundary the person is holding.
   */
  planRedo(node: string, opts?: { deferred?: boolean; commandCorrection?: boolean; whilePaused?: boolean | undefined }): RedoPlan {
    if (this.isPaused() && !opts?.whilePaused) throw new Error(`run paused before ${this.pausedAt().join(', ')}; explicitly unpause before reopening work`)
    const body = this.nodes.get(node)
    if (body === undefined) throw new Error(`unknown node ${node}`)
    if (body._type === 'GateNode') {
      // Gates are command-correctable when the correction targets the gate's evaluator.
      if (!opts?.commandCorrection) {
        // Find the agent whose edge leads into this gate.
        const incoming = this.edges.filter(edge => edge.target === node && edge.condition == null)
        const upstream = incoming.length === 1 ? incoming[0]!.source : undefined
        if (upstream !== undefined) {
          throw new Error(
            `${node} is a gate; replay ${upstream} instead, which re-runs ${upstream}, this gate and what follows, not the loop body`
          )
        }
        throw new Error(
          `${node} is a gate; replay the upstream agent instead, which re-runs that agent, this gate and what follows, not the loop body`
        )
      }
    }
    if (body._type === 'DataNode') {
      throw new Error(`node ${node} iterates its items; redo a completed node in its body or upstream`)
    }
    if (!this.completed.includes(node)) {
      if (!opts?.commandCorrection) {
        throw new Error(`node ${node} has not completed, so there is nothing to redo`)
      }
      // Command correction: the node is due (in cursor) or was never completed.
      // Accept it: plan the forward invalidation from this node.
      if (!this.cursor.includes(node)) {
        throw new Error(
          `node ${node} is neither completed nor due; command correction requires a due or completed node`
        )
      }
    }
    if (this.hasConcurrentWork()) {
      if (!opts?.deferred && !opts?.commandCorrection) {
        throw new Error('redo cannot discard concurrent branches; finish the fork before reopening completed work')
      }
    }
    const owners = [...this.nodes.entries()].filter(([, candidate]) =>
      candidate._type === 'DataNode' && candidate.subgraph_entry !== undefined
      && this.descendants(candidate.subgraph_entry, candidate.subgraph_exit).has(node))
    if (owners.length > 1) throw new Error(`node ${node} belongs to more than one data body; redo an unambiguous upstream node`)
    const owner = owners[0]
    const currentItem = owner !== undefined && this.pass?.node === owner[0]
    // A finished dataset no longer has a current item. Reopening a body node
    // therefore restarts its owning dataset, rather than running an orphan pass.
    const start = owner !== undefined && !currentItem ? owner[0] : node
    const affected = this.descendants(start, currentItem ? this.pass?.exit : undefined, { skipRetryEdges: true, followContinuations: true })
    if (currentItem && owner !== undefined) {
      for (const edge of this.outgoing(owner[0])) {
        for (const id of this.descendants(edge.target, undefined, { skipRetryEdges: true, followContinuations: true })) affected.add(id)
      }
    }
    const resetPass = this.pass !== undefined && affected.has(this.pass.node)
    if (this.pass !== undefined && !currentItem && !resetPass) {
      throw new Error(`redo of ${node} would abandon the unrelated active data pass ${this.pass.node}`)
    }
    return { requested: node, start, nodes: [...affected], resetPass,
      restartData: [...affected].filter(id => this.nodes.get(id)?._type === 'DataNode') }
  }

  /** Reopen a due node with a command correction and invalidate forward descendants; `whilePaused` as for {@link planRedo}. */
  redoCommandCorrection(node: string, opts?: { whilePaused?: boolean }): SpineNext {
    const plan = this.planRedo(node, { commandCorrection: true, whilePaused: opts?.whilePaused })
    const affected = new Set(plan.nodes)
    this.invalidateNodes(affected, true, false)
    if (plan.resetPass) this.pass = undefined
    else if (this.pass !== undefined) this.pass.finished = false
    this.cursor = this.cursor.filter(id => !affected.has(id))
    for (const [container, continuation] of this.afters) {
      if (affected.has(continuation)) this.afters.delete(container)
    }
    this.halted = undefined
    if (!this.cursor.includes(plan.start)) this.cursor.unshift(plan.start)
    const rearms: Array<{ container: string; continuation: string }> = []
    const seen = new Set<string>()
    for (const n of plan.nodes) {
      const cont = this.continuationOf(n)
      if (cont !== undefined && affected.has(cont.continuation) && !affected.has(cont.container)
          && cont.continuation !== plan.start && !this.cursor.includes(cont.continuation) && !seen.has(cont.continuation)) {
        seen.add(cont.continuation)
        rearms.push(cont)
      }
    }
    for (const { container, continuation } of rearms) {
      this.afters.set(container, continuation)
    }
    this.log.push(`command correction redid ${node} at ${plan.start}; invalidated ${plan.nodes.join(', ')}`)
    return this.next()
  }

  /** Reopen work and invalidate descendants; prompts and historical log remain. `whilePaused` as for {@link planRedo}. */
  redo(node: string, opts?: { preserveIterations?: boolean; whilePaused?: boolean }): SpineNext {
    const plan = this.planRedo(node, { whilePaused: opts?.whilePaused })
    const affected = new Set(plan.nodes)
    this.invalidateNodes(affected, true, opts?.preserveIterations ?? false)
    if (plan.resetPass) this.pass = undefined
    else if (this.pass !== undefined) this.pass.finished = false
    // Keep cursor entries and afters not in the affected set
    this.cursor = this.cursor.filter(id => !affected.has(id))
    for (const [container, continuation] of this.afters) {
      if (affected.has(continuation)) this.afters.delete(container)
    }
    this.halted = undefined
    // Add the redo start if not already present
    if (!this.cursor.includes(plan.start)) this.cursor.unshift(plan.start)
    // Re-arm the afters barrier for continuations whose container is not being
    // redone — the container's completeDue would normally push the continuation,
    // but it won't re-run.
    const rearms: Array<{ container: string; continuation: string }> = []
    const seen = new Set<string>()
    for (const n of plan.nodes) {
      const cont = this.continuationOf(n)
      if (cont !== undefined && affected.has(cont.continuation) && !affected.has(cont.container)
          && cont.continuation !== plan.start && !this.cursor.includes(cont.continuation) && !seen.has(cont.continuation)) {
        seen.add(cont.continuation)
        rearms.push(cont)
      }
    }
    for (const { container, continuation } of rearms) {
      this.afters.set(container, continuation)
    }
    this.log.push(`redid ${node} at ${plan.start}; invalidated ${plan.nodes.join(', ')}`)
    return this.next()
  }

  /**
   * Amend one node's prompt, so a re-run of that node uses the corrected task.
   *
   * Only an `AgentNode` or `LLMNode` carries a prompt. The amendment writes to
   * whichever field that node's view reports, so {@link view} carries the
   * correction with no further bookkeeping. `set` replaces the prompt; `append`
   * extends it on its own paragraph, and equals `set` when there is no prompt yet.
   * @param node - the node id whose prompt is amended.
   * @param text - the amendment text; must not be empty.
   * @param mode - `set` replaces the prompt, `append` adds a paragraph to it.
   * @returns nothing; the amended prompt is reported by {@link state} and {@link view}.
   */
  amendPrompt(node: string, text: string, mode: PromptAmendmentMode): void {
    const body = this.nodes.get(node)
    if (body === undefined) throw new Error(`unknown node ${node}`)
    const field = promptField(body)
    if (field === undefined) {
      throw new Error(`node ${node} is a ${body._type} and carries no prompt`)
    }
    if (text.trim() === '') throw new Error('prompt amendment text is required')
    const existing = promptText(body)
    // Guard: a set-prompt that drops {{...}} placeholders is refused (dm-159).
    if (mode === 'set') {
      // Strip the steering block so a brief containing {{...}} does not trigger
      // false refusals (agent_delivery ~541-549 does the same).
      const STEERING_START = '<!-- refactory steering context -->'
      const STEERING_END = '<!-- end refactory steering context -->'
      let scanText = existing
      const stStart = scanText.indexOf(STEERING_START)
      const stEnd = stStart >= 0 ? scanText.indexOf(STEERING_END, stStart) : -1
      if (stStart >= 0 && stEnd >= stStart) {
        scanText = scanText.slice(0, stStart) + scanText.slice(stEnd + STEERING_END.length)
      }
      const keys: string[] = []
      const re = /\{\{\s*([A-Za-z_]\w*)\s*\}\}/g
      let m: RegExpExecArray | null
      while ((m = re.exec(scanText)) !== null) keys.push(m[1] as string)
      if (keys.length > 0) {
        const missing = keys.filter(k => !new RegExp(`\\{\\{\\s*${k}\\s*\\}\\}`).test(text))
        if (missing.length > 0) {
          throw new Error(
            `set-prompt refused: the replacement drops ${missing.map(k => `{{${k}}}`).join(', ')}. `
            + `Use append-prompt to add your correction while keeping the template's placeholders.`
          )
        }
      }
    }
    body[field] = mode === 'append' && existing !== '' ? `${existing}\n\n${text}` : text
    this.promptEdits.add(node)
    this.log.push(`prompt ${node} ${mode}`)
  }

  /** Stop at a graph boundary after owned in-flight work has returned. */
  stopAtBoundary(): void { this.halted = { gate: this.cursor[0] ?? '', reason: 'cancelled' } }

  /**
   * The gate reason for a node: the latest refusal reason from any gate whose
   * retry edge targets this node, or undefined when no gate holds a reason for it.
   *
   * When more than one gate holds a reason for the node (e.g. both `check` and
   * `gate_qa` send work back to `emit`), the latest refusal is returned.
   */
  gateReasonFor(node: string): string | undefined {
    const retrySet = this.retryEdges()
    let best: { reason: string; at: number } | undefined
    for (const [gate, entry] of Object.entries(this.gateReasons)) {
      if (entry.reason === '') continue
      for (const edge of this.outgoing(gate)) {
        if (edge.target === node && retrySet.has(retryKey(gate, edge.target, edge.condition))) {
          if (best === undefined || entry.at > best.at) best = entry
        }
      }
    }
    return best?.reason
  }

  /** Serialize the complete cursor, including nested branches and graph edits. */
  checkpoint(): Record<string, unknown> {
    const redo_plans: Record<string, RedoPlan> = {}
    for (const node of this.completed) {
      // A person may preview and rerun behind their own pause, so the plans are
      // described while paused too (dm-227); the CEO's redo still refuses there.
      try { redo_plans[node] = this.planRedo(node, { whilePaused: true }) } catch { /* unsafe replay stays unavailable */ }
    }
    const deferred_redo_plans: Record<string, RedoPlan> = {}
    for (const node of this.completed) {
      if (node in redo_plans) continue
      try { deferred_redo_plans[node] = this.planRedo(node, { deferred: true, whilePaused: true }) } catch { /* unsafe replay stays unavailable */ }
    }
    return { schema_version: 2, source_graph: this.sourceGraph, state: this.state(),
      graph: { name: this.workflow, project_dir: this.projectDir, start_node: this.cursor[0] ?? '',
        nodes: Object.fromEntries(this.nodes), edges: this.edges },
      afters_map: Object.fromEntries(this.afters), successors: { ...this.successors }, redo_plans, deferred_redo_plans }
  }

  /** Restore only an explicit checkpoint of this exact source graph. */
  restore(value: Record<string, unknown>): void {
    if ((value.schema_version !== 1 && value.schema_version !== 2) || value.source_graph !== this.sourceGraph) {
      throw new Error('checkpoint inputs or source graph changed; explicit invalidation is required')
    }
    const saved = value.state as ReturnType<Spine['state']>
    const graph = value.graph as WorkflowGraph
    const list = (entry: unknown): string[] => {
      if (!Array.isArray(entry) || !entry.every(item => typeof item === 'string')) throw new Error('invalid checkpoint node list')
      return [...entry] as string[]
    }
    const cursor = list(saved.cursor), completed = list(saved.completed)
    if ([...cursor, ...completed].some(id => graph.nodes[id] === undefined)) throw new Error('checkpoint references an unknown node')
    this.nodes.clear()
    for (const [id, node] of Object.entries(graph.nodes)) this.nodes.set(id, node)
    this.edges = graph.edges
    this.cursor = cursor
    this.completed = completed
    this.skipped = list(saved.skipped)
    // Restore skip provenance from checkpoint (backward-compatible with older checkpoints).
    this.skipProvenance.clear()
    if (saved.skip_provenance != null && typeof saved.skip_provenance === 'object') {
      for (const [id, prov] of Object.entries(saved.skip_provenance as Record<string, { source: string; at: string }>)) {
        this.skipProvenance.set(id, prov)
      }
    }
    // Restore pending skip receipts (backward-compatible).
    this.pendingSkipReceipts = Array.isArray(saved.pending_skip_receipts) ? [...saved.pending_skip_receipts] : []
    this.pauseBoundaries.clear()
    for (const node of list(saved.pause_before)) this.pauseBoundaries.add(node)
    this.gateVerdicts.clear()
    for (const [id, verdict] of Object.entries(saved.gate_verdicts as Record<string, string>)) this.gateVerdicts.set(id, verdict)
    Object.assign(this.iterations, saved.iterations)
    Object.assign(this.feedback, saved.feedback)
    for (const key of Object.keys(this.gateReasons)) delete this.gateReasons[key]
    Object.assign(this.gateReasons, saved.gate_reasons ?? {})
    this.gateReasonSeq = Object.values(this.gateReasons as Record<string, { at?: number }>).reduce(
      (max, entry) => Math.max(max, typeof entry?.at === 'number' ? entry.at : 0), 0)
    Object.assign(this.successors, value.successors)
    this.log.splice(0, this.log.length, ...list(saved.log))
    this.promptEdits.clear()
    for (const node of Object.keys(saved.prompts as object)) this.promptEdits.add(node)
    this.pass = saved.pass as DataPass | undefined
    this.halted = saved.halted as TerminalHalt | undefined
    // Restore afters: v2 uses afters_map, v1 uses flat afters array (migration)
    if (value.schema_version === 2) {
      const map = value.afters_map as Record<string, string>
      this.afters = new Map(Object.entries(map))
    } else {
      // v1: flat array — reconstruct keyed map by matching each entry to its container
      const flatAfters = list(value.afters)
      this.afters = new Map()
      for (const continuation of flatAfters) {
        // Find the container whose continuation matches this entry
        const containers: Array<[string, string | undefined]> = [
          ...[...this.nodes].filter(([, body]) => body._type === 'ForkNode')
            .map(([id]): [string, string | undefined] => [id, this.unconditionalTarget(id)]),
          ...Object.entries(this.successors),
        ]
        for (const [container, cont] of containers) {
          if (cont === continuation) {
            this.afters.set(container, continuation)
            break
          }
        }
      }
    }
  }

  /** Full mutable state (for `wf_state`). */
  state(): Record<string, unknown> {
    const prompts: Record<string, string> = {}
    for (const node of this.promptEdits) {
      const body = this.nodes.get(node)
      if (body !== undefined) prompts[node] = promptText(body)
    }
    return {
      workflow: this.workflow,
      project_dir: this.projectDir,
      cursor: [...this.cursor],
      completed: [...this.completed],
      skipped: [...this.skipped],
      skip_provenance: Object.fromEntries(this.skipProvenance),
      pending_skip_receipts: [...this.pendingSkipReceipts],
      pause_before: [...this.pauseBoundaries],
      gate_verdicts: Object.fromEntries(this.gateVerdicts),
      iterations: { ...this.iterations },
      feedback: { ...this.feedback },
      gate_reasons: { ...this.gateReasons },
      afters: [...this.afters.values()],
      afters_map: Object.fromEntries(this.afters),
      prompts,
      ...(this.pass === undefined ? {} : { pass: { ...this.pass } }),
      ...(this.halted === undefined ? {} : { halted: { ...this.halted } }),
      log: [...this.log],
    }
  }
}
