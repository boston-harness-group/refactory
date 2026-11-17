/** A sealed step's tool list over DSH's scoped registry: those tools, and no other. */
export const inject = ['tools']

export function apply(ctx, config) {
  const allow = [...(config?.allow ?? [])]
  if (allow.length === 0) throw new Error('refactory-tools: a sealed step lists at least one tool')
  const allowed = new Set(allow)
  // Guards are monotonic: no permission hook or late registration can make another
  // tool executable in this worker, whichever agent calls it.
  ctx.tools.guard((exec) => !allowed.has(exec.name)
    ? `This step may use only ${allow.join(', ')}`
    // Its commands run sealed; a request to run one with wider access is refused here
    // rather than left to the approval policy.
    : exec.arguments?.sandbox_permissions !== undefined
      ? 'This step\'s commands run sealed; sandbox_permissions is refused' : undefined)
  ctx.on('agent/created', ({ agent }) => {
    agent.ctx.tools.restrict({ allow })
  })
}
