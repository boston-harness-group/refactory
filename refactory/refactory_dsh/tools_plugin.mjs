/** A sealed step's tool list over DSH's scoped registry: those tools, and no other. */
export const inject = ['tools']

// The arguments each tool may be called with. Its commands run sealed; an argument
// beyond these (sandbox_permissions asking for wider access, run_in_background) is
// refused, and a tool with no list here is never admitted.
const ARGUMENTS = { bash: ['command', 'description', 'timeoutMs', 'workdir'] }

export function apply(ctx, config) {
  const allow = [...(config?.allow ?? [])]
  if (allow.length === 0) throw new Error('refactory-tools: a sealed step lists at least one tool')
  const unknown = allow.filter(name => !ARGUMENTS[name])
  if (unknown.length) throw new Error(`refactory-tools: no argument list for ${unknown.join(', ')}`)
  const allowed = new Set(allow)
  // Guards are monotonic: no permission hook or late registration can make another
  // tool executable in this worker, whichever agent calls it.
  ctx.tools.guard((exec) => {
    if (!allowed.has(exec.name)) return `This step may use only ${allow.join(', ')}`
    const extra = Object.keys(exec.arguments ?? {}).filter(key => !ARGUMENTS[exec.name].includes(key))
    return extra.length ? `This step's ${exec.name} takes only ${ARGUMENTS[exec.name].join(', ')}; `
      + `refused: ${extra.join(', ')}` : undefined
  })
  ctx.on('agent/created', ({ agent }) => {
    agent.ctx.tools.restrict({ allow })
  })
}
