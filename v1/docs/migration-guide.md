# Migration Guide: `akashgit/remote-factory` → `boston-harness-group/remote-factory`

**Date:** 2026-08-22
**Owner:** Maintainers

This document covers what changes with the repo transfer and what each collaborator needs to do.

---

## What happens automatically

GitHub creates a **permanent redirect** from the old URL to the new one. This means:

- Existing `git clone` / `git pull` / `git push` commands continue to work via redirect
- Old links to issues, PRs, code, and raw files redirect to the new location
- Forks retain their upstream relationship
- Stars, watchers, and discussions transfer intact
- All open PRs (#1160–#1379) transfer with their review state preserved

**The redirect breaks if a new repo named `remote-factory` is ever created under `akashgit`.** Don't do that.

---

## What every collaborator must do

### 1. Update your git remote (recommended, not urgent)

The redirect works indefinitely, but updating avoids a round-trip on every fetch/push:

```bash
cd /path/to/remote-factory
git remote set-url origin git@github.com:boston-harness-group/remote-factory.git

# Verify
git remote -v
# Should show: boston-harness-group/remote-factory.git
```

If you use HTTPS instead of SSH:
```bash
git remote set-url origin https://github.com/boston-harness-group/remote-factory.git
```

### 2. Update local `factory` installs

If you installed via `uv tool install`:
```bash
uv tool install git+https://github.com/boston-harness-group/remote-factory.git
```

If you installed in dev/editable mode, no action needed — your local clone still works.

### 3. Accept the org invitation

GitHub will send you an email to join `boston-harness-group`. **You must accept the invitation to retain push access.** Until you accept, you can still read and clone, but you cannot push or review PRs.

Check pending invitations at: https://github.com/orgs/boston-harness-group/invitation

---

## What changes for open PR authors

**Your open PRs transfer automatically.** No action needed — the PR number, branch, and review state are preserved.

If your PR's branch is on a personal fork, it continues to work. If you had a local branch tracking the old remote, update the remote URL (step 1 above) and your branch will push to the right place.

### Open PRs at time of transfer

| PR | Author | Title |
|----|--------|-------|
| #1379 | @beatsmonster | fix(ci): stop the runtime image build failing on the digest filename |
| #1378 | @crqu | test: add coverage for venv injection in runners and eval runner |
| #1376 | @osilkin98 | fix: correct 3 test bugs from workflow migration |
| #1373 | @xukai92 | feat: add vi mode detection for tmux send-keys |
| #1364 | @xukai92 | Exp #36: Harden tmux key submission |
| #1361 | @lukeinglis | feat: show in-progress candidates in outer-loop status |
| #1341 | @itsnevu | Raise instead of guessing main |
| #1324 | @crqu | fix: override strategist post_checks in design_workflow |
| #1321 | @mihirathale98 | fix: clean up stale modes and fix auto-routing |
| #1319 | @colehurwitz | feat: TDD pipeline |
| #1302 | @akashgit | Dead-code pruning report |
| #1290 | @lukeinglis | docs: add custom endpoint profile documentation |
| #1289 | @lambdabaa | experiment: pydantic-graph prototype |
| #1286 | @mihirathale98 | fix: specify explicit graph.json path |
| #1275 | @akashgit | feat: builder-only seed + mixed calibration |
| #1273 | @akashgit | fix: Outer Loop v1 Post-Mortem |
| #1271 | @osilkin98 | feat(skillopt): add meta-skill module |
| #1268 | @shivchander | fix: add user-global tier to agent prompt resolution |
| #1267 | @shivchander | fix: allow plugin-registered agent roles |
| #1265 | @shivchander | fix: carry plugin-created .factory/ subdirs into worktrees |
| #1261 | @akashgit | feat: add outer loop Phase 3 — Designer Agent |
| #1258 | @akashgit | feat: outer loop Phase 1 |
| #1255 | @abhi1092 | feat: add --focus flag to workflow run |
| #1224 | @xukai92 | feat: replace OptimizationLoop with optimize workflow graph |
| #1218 | @colehurwitz | feat: open agent roles |
| #1208 | @beatsmonster | fix(contained): verify/setup UX |
| #1193 | @Maxusmusti | feat: add factory/compress/ package |
| #1173 | @RobotSail | feat: add workflow plugin system |
| #1162 | @xukai92 | feat: add benchmark split protocol |
| #1160 | @xukai92 | feat: add dynamic benchmark loader |

---

## What the maintainers will fix post-transfer

A cleanup PR will update all hardcoded references. **Collaborators do not need to do this** — it will land as a single PR after the transfer.

### Codebase references (`akashgit/remote-factory` → `boston-harness-group/remote-factory`)

| File | What changes |
|------|-------------|
| `pyproject.toml` | Homepage, Repository, Issues URLs |
| `README.md` | Logo URL, license badge, install command, Langfuse link, plugin marketplace |
| `mkdocs.yml` | `repo_url`, `repo_name` |
| `install.sh` | `REPO_URL` clone target |
| `docs/setup.md` | Install commands (curl, uv, git clone) |
| `docs/index.md` | License badge URL |
| `docs/benchmarks.md` | `REPO` constant in JS |
| `docs/full-eval.md` | `FE_REPO` constant in JS |
| `docs/contributing.md` | `git clone` URL |
| `docs/contributing-benchmarks.md` | 6 documentation cross-links |
| `docs/contained/index.md` | 4 GHCR image references (docs only) |
| `benchmarks/factory_harbor_agent.py` | `uv tool install` URL |
| `.github/workflows/benchmark.yml` | Install URL + 4 doc links in PR body |
| `.claude-plugin/plugin.json` | `homepage`, `repository` |
| `.codex-plugin/plugin.json` | `homepage`, `repository` |
| `skills/implement/SKILL.md` | Install URL |
| `factory/contained/setup.py` | Clone URL in setup output |
| `scripts/review-prs.sh` | `REPOS` array |
| `scripts/langfuse/README.md` | Issue link |

### Container image (GHCR)

The runtime image currently lives at:
```
ghcr.io/akashgit/remote-factory/factory-runtime:latest
```

It will move to:
```
ghcr.io/boston-harness-group/remote-factory/factory-runtime:latest
```

The CI workflow (`.github/workflows/runtime-image.yml`) uses `${{ github.repository }}`, so it will automatically push to the new path on the next build. The hardcoded default in `factory/podman.py:40` will be updated in the cleanup PR.

**If you use `factory contained`:** After the cleanup PR lands, run `factory contained setup` to pull the new image. Until then, the old image continues to work locally — it's already pulled.

### GitHub Pages

The docs site URL changes:
- **Old:** `https://akashgit.github.io/remote-factory/`
- **New:** `https://boston-harness-group.github.io/remote-factory/`

The old URL will stop working. The new URL requires GitHub Pages to be enabled in the org settings (the maintainers will do this).

### Downstream dependency: `refactory-midstream`

The `refactory-midstream` repo has a git dependency on `remote-factory` in its `pyproject.toml`:
```toml
remote-factory = { git = "https://github.com/akashgit/remote-factory.git" }
```
This will continue to work via redirect, but will be updated separately.

---

## What does NOT change

- The package name remains `remote-factory` (PyPI name, `factory` CLI command)
- The `.factory/` directory layout in target projects is unchanged
- Agent prompts, workflow definitions, and skills are unaffected
- Your `~/.factory/config.toml` and `~/.factory/registry.json` need no changes
- The MIT license and all contributor attribution are preserved

---

## Timeline

1. **Transfer happens:** Immediate redirect active, access requires org invitation acceptance
2. **Cleanup PR lands:** Within 24 hours — updates all hardcoded references
3. **Container image rebuilt:** Automatic on next push to `main` after cleanup PR
4. **Old redirect:** Works indefinitely (unless a new repo named `remote-factory` is created under the old owner)

---

## Questions?

Reach out in the repo Discussions or open an issue.
