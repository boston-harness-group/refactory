import tsconfigPaths from 'vite-tsconfig-paths'
import { defineConfig } from 'vitest/config'
import { standardDecoratorPlugin, vitestExecArgv } from './vitest.shared.ts'

// Resolution facade shared by every plugin instance below: tsconfig.base.json
// has no include, which vite-tsconfig-paths treats as match-all, so its paths map
// applies to every test file. The map covers the eight packages this fork owns;
// every harness specifier resolves through node_modules to the published pin.
/** Spec files that start many processes at once (see `projects` below). */
const SPAWNING = ['packages/workflow/workflow-spine/tests/data-parallel.spec.ts']

const pathsPlugin = (): ReturnType<typeof tsconfigPaths> => tsconfigPaths({ projects: ['./tsconfig.base.json'] })

export default defineConfig({
  plugins: [pathsPlugin(), standardDecoratorPlugin()],
  test: {
    execArgv: vitestExecArgv,
    // The published harness packages ship bundled JavaScript that imports CSS
    // and undeclared third-party libraries; inlining them lets Vite apply the
    // CSS pipeline and resolve those imports from this workspace's root.
    server: { deps: { inline: [/@deepseek-ai\//] } },
    // Forked workers: Node 24 has aborted in its CJS lexer from worker threads.
    globalSetup: ['./vitest.global-setup.ts'],
    pool: 'forks',
    // Specs that start many processes at once run as a group of their own, after the
    // rest: data-parallel.spec.ts runs items four at a time, each a real Python process
    // (about 240 in all), which on a 4-vCPU runner starved the cheap specs running beside
    // it past their 5 s limit (a 20 ms test took 8 s). Together they finish no later, and
    // no spec shares the machine with that burst.
    projects: [
      // .tsx: client component specs (jsdom via per-file @vitest-environment pragma).
      { extends: true, test: { name: 'specs', include: ['packages/*/*/tests/**/*.spec.{ts,tsx}'],
                               exclude: [...SPAWNING, '**/node_modules/**'] } },
      { extends: true, test: { name: 'spawning', include: SPAWNING, sequence: { groupOrder: 1 } } },
    ],
    coverage: {
      provider: 'v8',
      include: ['packages/*/*/src/**/*.{ts,tsx}'],
      reporter: ['text'],
    },
  },
})
