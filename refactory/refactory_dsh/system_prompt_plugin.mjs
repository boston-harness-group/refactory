/**
 * DSH composition plugin that injects the node's rendered system-prompt text
 * as a system-prompt section.  Reads `REFACTORY_SYSTEM_PROMPT` from the
 * environment and calls `ctx.systemPrompt.section()` with it.
 *
 * The plugin is a no-op when the env var is empty or unset.
 */

export const inject = ['systemPrompt'];

export function apply(ctx) {
  const systemText = process.env.REFACTORY_SYSTEM_PROMPT;
  if (!systemText) return;
  ctx.systemPrompt.section({
    name: 'refactory-node-system',
    order: ctx.systemPrompt.getSectionOrder('TOOL_WORKFLOW') + 1,
    text: systemText,
  });
}
