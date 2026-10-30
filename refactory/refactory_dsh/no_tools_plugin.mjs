/** Strict no-tools policy over DSH's actual scoped registry and provider seam. */
import { readFileSync, writeFileSync } from 'node:fs'

export const inject = ['tools', 'llm', 'agents']

export function apply(ctx) {
  // Guards are monotonic: neither a permission hook nor a late registration can
  // make an unsolicited model tool call executable in this worker.
  ctx.tools.guard(() => 'This agent declares tool_policy=none')
  ctx.on('agent/created', ({ agent }) => {
    agent.ctx.tools.restrict({ allow: [] })
  })
  ctx.on('llm/stream', (options, next) => {
    // Scope-local tools can bypass inherited restrictions. Refuse such a
    // composition before a provider sees a request rather than silently leak.
    if (options.tools?.length) {
      throw new Error('tool_policy=none cannot expose scoped tools to the provider')
    }
    const path = process.env.REFACTORY_WORKER_RECEIPT
    if (path) {
      const receipt = JSON.parse(readFileSync(path, 'utf8'))
      writeFileSync(path, JSON.stringify({ ...receipt,
        enforced_tool_policy: 'none', tools_permitted: [],
        tool_observation_available: true, tool_policy_source: 'dsh-provider-request',
      }, null, 2) + '\n')
    }
    return next()
  })
}
