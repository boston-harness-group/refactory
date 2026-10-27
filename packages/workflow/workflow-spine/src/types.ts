/**
 * Graph vocabulary for the workflow spine. Field names deliberately match
 * refactory's `Workflow.to_dict()` output (snake_case) — that JSON is the wire
 * format this package consumes.
 * @module @deepseek-ai/dsh-workflow-spine/types
 */

/** A director annotation on an AgentNode: hot-plug N child agents from a plan. */
export interface DirectorExpansion {
  /** Artifact path the director writes (the plan). */
  plan: string
  /** Child agent role. */
  role: string
  /** Plan-entry field holding the child slug. */
  slug_field: string
  /** Plan-entry field holding the child prompt. */
  prompt_field: string
  /** Child writes template, with `{slug}` substituted. */
  writes_template: string
}

/** One workflow node. Loose on purpose: it mirrors an external JSON format. */
export interface WorkflowNode {
  /** The node's IR type, e.g. `AgentNode` or `FnNode`. Selects the behavior. */
  _type: string
  /** The node id the graph keys it by, and the id its commands are told. */
  id: string
  /** `AgentNode`: the agent role the node is declared with. */
  role?: string
  resource_role?: 'work' | 'final_validation'
  /** The artifacts the node produces, as paths relative to the run directory. */
  writes?: string[]
  /** The artifacts the node consumes; a subagent is told to read these. */
  reads?: string[]
  /** `AgentNode`: the task handed to the driving agent, with `{{fact}}` slots. */
  prompt_template?: string
  /** `LLMNode`: the system prompt for the one-shot call. */
  system_prompt?: string
  /** `LLMNode`: an alternative per-instance prompt when no system prompt is set. */
  instance_prompt?: string
  /** `AgentNode` / `LLMNode`: the model the node was declared with. */
  model?: string
  /** `LLMNode`: sampling temperature. */
  temperature?: number
  /** `LLMNode`: the node's own turn cap. */
  max_turns?: number
  /** `GateNode`: the cap for this gate's reloop edge. */
  max_iterations?: number
  /** `FnNode` / `Study`: prose the driving agent follows when it runs the node. */
  content?: string
  /** `FnNode` / `Study`: the shell command the spine runs for the node. */
  command?: string
  /** `FnNode` / function gate: resolved envelope for an operation with bounded passes. */
  command_timeout_s?: number
  /** `GateNode`: who answers it — its evaluator, a person, or the driving agent. */
  evaluator_type?: 'fn' | 'user' | 'agent'
  /** `GateNode` with `evaluator_type` `fn`: the command whose last stdout line is the verdict, and any lines before it the reason. */
  evaluator_command?: string
  /** `GateNode`: what the person or the driving agent is asked. */
  gate_prompt?: string
  /** `ForkNode`: the branch entry nodes, which are reached through this list and not through edges. */
  targets?: string[]
  /** `ForkNode`: targets whose failure does not fail the fork. */
  optional_targets?: string[]
  /** `JoinNode`: the branch exits this barrier waits for. */
  sources?: string[]
  /** `AgentNode`: the annotation that lets a director hot-plug its planned children. */
  expand?: DirectorExpansion
  /** `DataNode`: the Task binding that supplies the items and judges each pass. */
  task_ref?: string
  /** `DataNode`: the item source read as `source_format`, when the items are not inline. */
  source_path?: string
  /** `DataNode`: how to read `source_path` — `directory`, `jsonl`, or `csv`. */
  source_format?: string
  /** `DataNode`: the items declared in the graph itself, each with an `id`. */
  inline_items?: Array<Record<string, unknown>>
  /** `DataNode`: the first node of the subgraph run once per item. */
  subgraph_entry?: string
  /** `DataNode`: the subgraph's last node, included in every pass. */
  subgraph_exit?: string
  /** `DataNode`: the most items to iterate, after the source is read. */
  limit?: number
  /** `SelectionNode`: the selection strategy. */
  strategy?: string
  /**
   * `AgentNode`: where the node works — a fresh scratch folder or a prepared clone of
   * the run's worktree — and how its output is collected. Python's worker
   * (`refactory_dsh/workspace.py`) prepares and collects it, and Python's lowering
   * refuses it on the CEO-subagent route; the spine only passes it through.
   */
  workspace?: AgentWorkspace
  [key: string]: unknown
}

/** An agent node's workspace declaration, as `refactory_dsh.nn.graph.workspace_declaration` writes it. */
export interface AgentWorkspace {
  kind: 'scratch' | 'clone'
  /** `clone`: a clone of the run's last attempt or of its base. */
  of?: 'attempt' | 'base'
  /** `files` copies the declared writes out; `diff` writes the clone's diff as the answer file. */
  collect: 'files' | 'diff'
}

/** One directed edge with an optional gate condition label. */
export interface WorkflowEdge {
  /** The node the walk leaves. */
  source: string
  /** The node the walk arrives at. */
  target: string
  /**
   * The verdict label that routes along this edge, or null/absent for the
   * unconditional edge. `proceed` / `reloop` / `halt` carry their usual
   * meaning. Other labels are forward outcomes unless a residual single-gate
   * cycle identifies a retry. Ambiguous multi-gate cycles must mark their retry
   * edges `reloop`. Labels are compared in lowercase.
   */
  condition?: string | null
}

/** A workflow graph — the exact shape of refactory's `Workflow.to_dict()`. */
export interface WorkflowGraph {
  /** The run's name, used in status and in the spine's own reporting. */
  name: string
  /** The node the cursor starts at. */
  start_node: string
  /** The directory the mode was lowered from, for a reader of the graph. */
  project_dir?: string
  /** Every node, keyed by the id it also declares. */
  nodes: Record<string, WorkflowNode>
  /** Every edge, with a gate's verdict labels as conditions. */
  edges: WorkflowEdge[]
}
