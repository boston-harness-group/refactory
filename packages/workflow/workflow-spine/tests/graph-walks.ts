/**
 * Independent graph-walking functions that compute expected invariant values
 * without using Spine methods. These are the test oracles.
 */
import type { WorkflowGraph } from '../src/types.ts'
import { retryEdges, retryKey } from '../src/loops.ts'

/** Compute the set of retry edges for a graph. */
export function computeRetryEdgeSet(graph: WorkflowGraph): Set<string> {
  return retryEdges(graph.nodes, graph.edges)
}

/**
 * Compute a fork's body: the nodes reachable from the fork's targets,
 * stopping at the fork's unconditional edge target (the join), skipping
 * retry back-edges. Does not use JoinNode.sources.
 */
export function forkBody(graph: WorkflowGraph, forkId: string, retrySet: Set<string>): Set<string> {
  const forkNode = graph.nodes[forkId]
  if (forkNode?._type !== 'ForkNode') return new Set()

  // The join is the fork's unconditional edge target
  const joinEdge = graph.edges.find(e => e.source === forkId && e.condition == null)
  const joinId = joinEdge?.target

  const body = new Set<string>()
  const targets = forkNode.targets ?? []
  const queue = [...targets]
  while (queue.length > 0) {
    const n = queue.shift()!
    if (n === joinId || body.has(n)) continue
    body.add(n)
    for (const e of graph.edges) {
      if (e.source === n && !retrySet.has(retryKey(e.source, e.target, e.condition))) {
        queue.push(e.target)
      }
    }
    const nd = graph.nodes[n]
    if (nd?._type === 'ForkNode') {
      for (const t of nd.targets ?? []) queue.push(t)
    }
    if (nd?._type === 'DataNode' && nd.subgraph_entry) {
      queue.push(nd.subgraph_entry)
    }
  }
  return body
}

/**
 * Independently compute graph sinks. A real sink is a node that has no
 * continuation — no outgoing edge that isn't a retry back-edge, no
 * fork-body containment, and not a data body exit.
 */
export function computeRealSinks(graph: WorkflowGraph, retrySet: Set<string>): string[] {
  // Data body exits: subgraph_exit of each DataNode
  const dataBodyExits = new Set<string>()
  for (const node of Object.values(graph.nodes)) {
    if (node._type === 'DataNode' && node.subgraph_exit) dataBodyExits.add(node.subgraph_exit)
  }

  // Nodes inside a fork body
  const insideForkBody = new Set<string>()
  for (const node of Object.values(graph.nodes)) {
    if (node._type !== 'ForkNode') continue
    const body = forkBody(graph, node.id, retrySet)
    for (const id of body) insideForkBody.add(id)
  }

  return Object.keys(graph.nodes).filter(id => {
    const hasOutgoing = graph.edges.some(e =>
      e.source === id && !retrySet.has(retryKey(e.source, e.target, e.condition)))
    if (hasOutgoing) return false
    if (dataBodyExits.has(id)) return false
    if (insideForkBody.has(id)) return false
    return true
  })
}

/**
 * For a node inside a fork body, return the innermost fork and its continuation
 * (the join). For a node inside a data body, return the data node's successor.
 */
export function containerContinuation(
  graph: WorkflowGraph, nodeId: string, retrySet: Set<string>,
  /** Director successor map from the live spine (populated after expand). */
  successors?: Record<string, string>,
): { container: string; continuation: string } | undefined {
  const candidates: Array<{ container: string; continuation: string; size: number }> = []

  // Check forks
  for (const node of Object.values(graph.nodes)) {
    if (node._type !== 'ForkNode') continue
    const joinEdge = graph.edges.find(e => e.source === node.id && e.condition == null)
    if (!joinEdge) continue
    const body = forkBody(graph, node.id, retrySet)
    if (body.has(nodeId)) {
      candidates.push({ container: node.id, continuation: joinEdge.target, size: body.size })
    }
  }

  // Check expanded directors (from successors map)
  if (successors) {
    for (const [directorId, successor] of Object.entries(successors)) {
      // Director children are id-prefixed with director__
      const prefix = `${directorId}__`
      // Check if nodeId is a child of this director
      const childEdges = graph.edges.filter(e => e.source === directorId && e.condition == null)
      const children = childEdges.map(e => e.target).filter(t => t.startsWith(prefix))
      if (children.includes(nodeId)) {
        candidates.push({ container: directorId, continuation: successor, size: children.length })
      }
    }
  }

  // Check data nodes
  for (const node of Object.values(graph.nodes)) {
    if (node._type !== 'DataNode' || !node.subgraph_entry || !node.subgraph_exit) continue
    // Compute data subgraph body
    const body = new Set<string>()
    const queue = [node.subgraph_entry]
    while (queue.length > 0) {
      const n = queue.shift()!
      if (body.has(n)) continue
      body.add(n)
      if (n === node.subgraph_exit) continue
      for (const e of graph.edges) {
        if (e.source === n && !retrySet.has(retryKey(e.source, e.target, e.condition))) {
          queue.push(e.target)
        }
      }
      // Follow fork targets so a fork's branches are part of the body.
      const nd = graph.nodes[n]
      if (nd?._type === 'ForkNode') {
        for (const t of nd.targets ?? []) queue.push(t)
      }
    }
    if (body.has(nodeId)) {
      // Data node's unconditional edge target is the continuation
      const contEdge = graph.edges.find(e => e.source === node.id && e.condition == null)
      if (contEdge) {
        candidates.push({ container: node.id, continuation: contEdge.target, size: body.size })
      }
    }
  }

  if (candidates.length === 0) return undefined
  // Return the innermost (smallest body)
  candidates.sort((a, b) => a.size - b.size)
  return { container: candidates[0]!.container, continuation: candidates[0]!.continuation }
}

/**
 * Compute the forward closure of a node: all nodes reachable from nodeId by
 * following edges (skipping retry back-edges), fork targets, data subgraph
 * entries, and container continuations. This is the independent redo-affected
 * set, computed without planRedo.
 *
 * Per correction (3): for a node inside the body of a finished data pass,
 * start at its data node (the closure is the data node's, including the data
 * node itself).
 *
 * Per correction (4): compute on the live graph (which has expansion children).
 */
export function forwardClosure(
  graph: WorkflowGraph,
  nodeId: string,
  retrySet: Set<string>,
  /** Director successor map from the live spine (populated after expand). */
  successors?: Record<string, string>,
): Set<string> {
  // Check if nodeId is inside a completed data pass body; if so, start at the data node
  let startId = nodeId
  for (const node of Object.values(graph.nodes)) {
    if (node._type !== 'DataNode' || !node.subgraph_entry || !node.subgraph_exit) continue
    const body = new Set<string>()
    const queue = [node.subgraph_entry]
    while (queue.length > 0) {
      const n = queue.shift()!
      if (body.has(n)) continue
      body.add(n)
      if (n === node.subgraph_exit) continue
      for (const e of graph.edges) {
        if (e.source === n && !retrySet.has(retryKey(e.source, e.target, e.condition))) {
          queue.push(e.target)
        }
      }
      // Follow fork targets so a fork's branches are part of the body.
      const nd = graph.nodes[n]
      if (nd?._type === 'ForkNode') {
        for (const t of nd.targets ?? []) queue.push(t)
      }
    }
    if (body.has(nodeId)) {
      startId = node.id
      break
    }
  }

  const closure = new Set<string>()
  const bfsQueue = [startId]

  while (bfsQueue.length > 0) {
    const n = bfsQueue.shift()!
    if (closure.has(n)) continue
    closure.add(n)

    // Follow non-retry outgoing edges
    for (const e of graph.edges) {
      if (e.source === n && !retrySet.has(retryKey(e.source, e.target, e.condition))) {
        bfsQueue.push(e.target)
      }
    }

    // Director successor edges
    if (successors && successors[n] !== undefined) {
      bfsQueue.push(successors[n]!)
    }

    const nd = graph.nodes[n]

    // Fork targets
    if (nd?._type === 'ForkNode') {
      for (const t of nd.targets ?? []) bfsQueue.push(t)
    }

    // Data subgraph entries
    if (nd?._type === 'DataNode' && nd.subgraph_entry) {
      bfsQueue.push(nd.subgraph_entry)
    }

    // Container continuation: if n is inside a fork body or director body,
    // include the container's continuation
    const cont = containerContinuation(graph, n, retrySet, successors)
    if (cont !== undefined) {
      bfsQueue.push(cont.continuation)
    }
  }

  return closure
}
