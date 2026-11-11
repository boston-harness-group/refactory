/** One node delegated through the installed DSH named-provider registry. */
import { randomUUID } from 'node:crypto'

export const name = 'refactory-worker-bridge'
/**
 * What the bridge prints when cleanup fails after the answer is already on stdout
 * (`model.TEARDOWN_WARNING`): a warning, not a failure, so the client keeps the answer.
 */
const TEARDOWN_WARNING = 'dsh: worker teardown failed after its answer:'
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
    // As the headless runner does: an empty task is refused, never sent to a model.
    if (task.trim() === '') throw new Error('a task is required on stdin or as the task argument')
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
      // The answer is written: cleanup that fails now (e.g. waiting for the worker's
      // process scope to empty, under load) cannot change it, so it is a warning and the
      // exit says what the worker did, not what its cleanup did. Re-asking the model
      // would discard a finished answer and repeat any work a tool session did.
      const finished = owned
      owned = undefined
      for (const dispose of [() => finished.dispose(), () => handle.dispose()]) {
        try {
          await dispose()
        } catch (error) {
          process.stderr.write(`${TEARDOWN_WARNING} ${error instanceof Error ? error.message : String(error)}\n`)
        }
      }
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
