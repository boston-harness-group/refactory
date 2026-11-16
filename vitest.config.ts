import tsconfigPaths from 'vite-tsconfig-paths'
import { defineConfig } from 'vitest/config'
import { standardDecoratorPlugin, vitestExecArgv } from './vitest.shared.ts'

// Resolution facade shared by every plugin instance below: tsconfig.base.json
// has no include, which vite-tsconfig-paths treats as match-all, so its paths map
// applies to every test file. The map covers the eight packages this fork owns;
// every harness specifier resolves through node_modules to the published pin.
const pathsPlugin = (): ReturnType<typeof tsconfigPaths> => tsconfigPaths({ projects: ['./tsconfig.base.json'] })

/** The workflow spine's specs (see `projects` below). */
const SPINE = 'packages/workflow/workflow-spine/tests/**/*.spec.{ts,tsx}'

/**
 * What every project runs with. The projects do not extend the root config: a project
 * that extends it runs the root's global setup again (and arrays such as `include`
 * merge), so the repository guard would run once per project instead of once.
 */
const shared = {
  plugins: [pathsPlugin(), standardDecoratorPlugin()],
  test: {
    execArgv: vitestExecArgv,
    // The published harness packages ship bundled JavaScript that imports CSS
    // and undeclared third-party libraries; inlining them lets Vite apply the
    // CSS pipeline and resolve those imports from this workspace's root.
    server: { deps: { inline: [/@deepseek-ai\//] } },
    // Forked workers: Node 24 has aborted in its CJS lexer from worker threads.
    pool: 'forks' as const,
  },
}

export default defineConfig({
  plugins: shared.plugins,
  test: {
    globalSetup: ['./vitest.global-setup.ts'],
    // Spine specs write checkpoints, status files and prompts on every step. On one CI
    // runner (run 43795023262, attempt 1) the spine specs that ran first were 20-35x
    // slower than in the re-run (data.spec 18 s vs 0.5 s) while in-memory specs running
    // beside them were not, and two ~20 ms tests passed the default 5 s: the machine, not
    // the tests. Spine specs get 30 s; the slow-test report (vitest.slow-reporter.ts)
    // names every test over 2 s, so a real slowdown still shows. No retries: a retry
    // would hide a real failure.
    projects: [
      { ...shared, test: { ...shared.test, name: 'spine', include: [SPINE], testTimeout: 30_000 } },
      // .tsx: client component specs (jsdom via per-file @vitest-environment pragma).
      { ...shared, test: { ...shared.test, name: 'packages',
                           include: ['packages/*/*/tests/**/*.spec.{ts,tsx}'],
                           exclude: [SPINE, '**/node_modules/**'] } },
    ],
    reporters: ['default', './vitest.slow-reporter.ts'],
    coverage: {
      provider: 'v8',
      include: ['packages/*/*/src/**/*.{ts,tsx}'],
      reporter: ['text'],
    },
  },
})
