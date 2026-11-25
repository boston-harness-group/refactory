<!-- CLAUDE.md imports this file; keep the guidance here. -->
# AGENTS.md

Guidance for coding agents (Codex, Claude Code and other AGENTS.md-aware tools)
working in this repository.

- Read [refactory/docs/contributing.md](refactory/docs/contributing.md) first: the
  commands, the invariants, where each kind of change goes, and the review checklist.
- Run only the tests your change touches while you iterate (`pnpm test:changed`), and the
  whole Python suite at most once before pushing: CI is the binding check. Never run the
  suite repeatedly, and never on a single worker.
- A new outer-loop method (a parent picker, a proposer, an outer loop, a port of a
  published optimizer) follows
  [Adding an optimization method](refactory/docs/methods.md): a package that composes
  the core building blocks and registers through `Package.optimizers` or
  `Package.strategies`.
- Reuse the core building blocks (parent pickers, proposers, `ValidationGate`,
  `FinalEvaluation`, the strategy trace, budgets). Do not write private copies; a
  genuinely new general piece goes in core once and the method imports it.
- Review checks this: a duplicated helper is a finding.

More guides (benchmark packages, edit operations, node kinds and judges) will follow.
