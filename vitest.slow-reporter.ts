/**
 * Names every test slower than SLOW_MS at the end of a run, slowest first, so a test that
 * got slow is seen even though the spine's time-out has headroom (vitest.config.ts).
 */
import type { Reporter, TestModule } from 'vitest/node'

const SLOW_MS = 2_000

export default class SlowTestReporter implements Reporter {
  onTestRunEnd(testModules: ReadonlyArray<TestModule>): void {
    const slow: Array<{ name: string; ms: number }> = []
    for (const module of testModules) {
      for (const test of module.children.allTests()) {
        const ms = test.diagnostic()?.duration ?? 0
        if (ms > SLOW_MS) slow.push({ name: `${module.moduleId.split('/packages/').pop()} > ${test.fullName}`, ms })
      }
    }
    if (slow.length === 0) return
    slow.sort((a, b) => b.ms - a.ms)
    console.log(`\nTests over ${SLOW_MS / 1000} s (${slow.length}):`)
    for (const { name, ms } of slow) console.log(`  ${(ms / 1000).toFixed(1).padStart(6)} s  ${name}`)
  }
}
