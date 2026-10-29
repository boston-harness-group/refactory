/** One node delegated through the installed DSH named-provider registry. */
import { randomUUID } from 'node:crypto'

export const name = 'refactory-worker-bridge'
export const inject = ['agents', 'subagents', 'headlessStartup']
export function apply(ctx) {
  const controller = new AbortController()
  let owned
  ctx.on('dispose', async () => {
    controller.abort()
    await owned?.dispose()
  })
  void (async () => {
    await ctx.get('loader')?.await()
    // The parent only owns the external worker, which answers on its own product's
    // settings (Codex, Claude Code); the parent itself never calls a model. It takes
    // the route the launch names, if any, and never the headless profile's default
    // model, so nothing here can reach a provider nobody chose (dm-258).
    const { provider, model } = ctx.get('headlessStartup')
    let task = ctx.get('headlessStartup').task
    // The launcher passes "-" to mean "read the prompt from stdin" (large prompts
    // exceed MAX_ARG_STRLEN).  The headless-runner resolves this, but the bridge
    // disables the headless-runner and must resolve it itself.
    if (task === undefined || task === '-') {
      const chunks = []
      for await (const chunk of process.stdin) chunks.push(chunk)
      task = Buffer.concat(chunks).toString('utf8')
    }
    const handle = await ctx.get('agents').create({
      sessionId: `refactory-parent-${randomUUID()}`,
      meta: { cwd: process.cwd() },
      agentOptions: provider && model ? { provider, model } : {},
    })
    try {
      owned = await ctx.get('subagents').start('refactory-worker', {
        parent: handle.agent,
        prompt: [{ type: 'text', text: task }],
        signal: controller.signal,
      })
      const result = await owned.result
      process.stdout.write(result.output.filter(block => block.type === 'text').map(block => block.text).join('') + '\n')
      if (result.stopReason !== 'completed') {
        process.stderr.write(`dsh: worker ${result.stopReason}: ${result.diagnostic ?? 'no diagnostic'}\n`)
      }
      await owned.dispose()
      owned = undefined
      await handle.dispose()
      ctx.get('appExit')(result.stopReason === 'completed' ? 0 : 1)
    } catch (error) {
      await owned?.dispose()
      owned = undefined
      await handle.dispose()
      throw error
    }
  })().catch(error => {
    // Provider diagnostics already exclude raw protocol/credential material.
    process.stderr.write(`dsh: worker failed: ${error instanceof Error ? error.message : String(error)}\n`)
    ctx.get('appExit')(1)
  })
}
