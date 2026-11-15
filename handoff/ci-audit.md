# CI audit and cleanup plan: boston-harness-group/refactory

The audit read origin/main at `df28750eb2` (#659). It covers 200 workflow runs from 2026-11-08 to 2026-11-15 (128 succeeded, 30 failed, 38 were cancelled because a newer push replaced them, 4 were still in progress). I read 39 failed attempts in full, including the failed first attempts of runs that passed on a re-run. Per-test times come from the `python-suite-times` JUnit files of runs 43700298262, 43692760565 and 43688137743. Nothing was pushed, re-run or edited.

## Summary

- **Where the time goes.** The Linux job takes a median of 26 min (range 25–40). Almost all of that is the **Python suite step: median 24.8 min (range 23.4–37.1)**. Install takes 0.4 min, the TypeScript suite 0.8 min and web acceptance 0.2 min. The macOS job takes 4–8 min and runs alongside it.
- **pytest already runs in parallel.** It uses `-n 4`, but `conftest.py:318-325` forces xdist's `loadscope` mode, which keeps each test file on one worker. Test time adds up to **114 min (median of 3 runs; range 92–140)**, so 4 workers end up at about 29 min. `test_delivery` alone is 18 min on a single worker, which is the floor for any plan that adds workers without splitting that file. The workflow comment ("22.4 min serial to 5.7 min at four on the lab") is out of date: a test here runs about 4x slower than on the lab.
- **Caching would not help.** Install is already 0.4 min. No build runs twice inside a job. Caching would save under a minute.
- **Red runs.** Of the 39 failed attempts, **23 were flaky or unrelated to the pull request** and 16 were real. 11 of the 16 real ones (7 stale-docs failures, 4 forgotten module-classification failures) could have been caught in seconds before pushing.
- **The worst flake has one root cause.** That is the `test_delivery` 'run warning' failure: 6 runs, 3 of them on main. Two test files write `refactory/tests/test_ops_parallel.py` into the **real checkout** while another worker's design run is watching that checkout.
- **One flake is a real product race.** `test_live_activity` torn lines (5 runs) is most likely a race in `live.emit`, not a test problem.
- **Expected CI time.** About 26–37 min today. About **10–12 min** after the sharding PR. About **8–10 min** after the trim PR. Every test still runs on every push.

---

## 1. Workflows (`.github/workflows/ci.yml`, the only workflow)

| job | steps | notes |
|---|---|---|
| `install-and-test` (ubuntu-latest, 4 vCPU, 60 min timeout) | checkout (fetch-depth 0) → pnpm/setup → setup-uv (py 3.12) → `./install.sh` → `pnpm test` → `pnpm run test:acceptance` → sandbox sysctl → `pytest refactory/tests -q -rs -n 4 --junitxml` → upload JUnit → harness_check | `install.sh` runs `uv sync`, `pnpm install --frozen-lockfile` and `pnpm run build`. `build` = `docs:check` + `typecheck` + 2× `tsdown`. Docs regen/check therefore runs on every push, through install.sh. |
| `macos` (macos-latest) | same install → `pnpm test` → 6 launch-budget/guard files `-n 3` | Needed for the platform. The TypeScript suite repeats here on purpose. |

Findings:

1. **There is no build-twice or docs-twice redundancy inside a job.** `pnpm test` (vitest) does not rebuild, and docs:check runs once per job. Repeating install on macOS is required. Install is 0.4 min, so caching is not worth adding (setup-uv already caches uv by default on hosted runners).
2. **pytest is parallel, but grouped by file:** `-n 4` plus the forced `--dist loadscope`. The comment in conftest says this exists because of `test_create` and `test_edit_design`, which share `.factory/packages` under the checkout and already take a file lock. Only 7 test files define module- or session-scoped fixtures. The ones that matter are `test_runtime_equivalence.runs` (5 design runs, 109 s) and `test_delivery`'s `_new_golden`/`_opened_golden`.
3. **There is no per-test timeout.** `pytest.mark.timeout` in `test_coverage_rerun_scripted.py:89,151` shows up as an "Unknown mark" warning, because pytest-timeout is not installed. A hung test would burn the job's full 60 minutes.
4. **`-rs` hides failures in the summary.** It replaces pytest's default `-rfE`, so no `FAILED` lines appear in the short summary. In a 1,500-line log, failures can only be found by searching for `____ name ____` headers.
5. **Wall time is spread unevenly** (92–140 min of summed test time for the same suite). Runner load matters, so any time target needs headroom.
6. **The local pre-push hook (lefthook) runs only `typecheck`.** It does not run `docs:check`, which is the most common real failure (7 runs).

## 2. Failures (39 failed attempts, 2026-11-08 to 11-15)

### 2a. Flaky or unrelated tests (23 attempts)

| # | test | failed in (run, branch/commit) | evidence it is flaky | likely cause | proposed fix |
|---|---|---|---|---|---|
| F1 | `test_delivery::test_no_pr_run_leaves_pr_text_and_title_in_delivery_json` (`:4117`), `test_no_pr_finish_commands_are_complete` (`:4364/4392`), `test_no_pr_keeps_the_work_on_its_branch_and_asks_github_nothing` (`:3233`) | 42514517137 search-combo f656b220 (passed on re-run); 42532697266 **main** 31b6d3e0; 43691614524 **main** d98743b8; 43695769021 **main** 25c59141; 43696326134 structure-ops 0a53f510; 43701146339 structure-ops f290c631 | Re-run passed; it fails on unrelated pull requests. Every one of the 6 messages reads `files outside its worktree changed: in /home/runner/work/refactory/refactory, uncommitted changes differ in refactory/tests/test_ops_parallel.py` | `test_data_parallel.py:156-162` (an autouse fixture, so all ~75 tests) and `test_evaluate_many.py:287-301` **write and delete `refactory/tests/test_ops_parallel.py` in the real checkout**. Each writes a *different* file body. A design run records its launcher checkout (pytest's working directory, which is the checkout) as `warn_also` (`repo_change.py:373-379`). `target_guard._unwatched()` (`REFACTORY_GUARD_UNWATCHED`, set in `conftest.py:86`) is applied in `checkouts()` but **not to `warn_also`**. So a write on another xdist worker turns into a "run warning" on stderr. | (a) Commit the two op modules as static helper files with distinct names (for example `tests/_ops_data_parallel.py` and `tests/_ops_evaluate_many.py`, which pytest will not collect). The fork-server children already import from `tests/`, so nothing is written at runtime. (b) Add a session check in `conftest.py`: `git status --porcelain` of `REPO_ROOT` must be the same at session end as at start, so any future test that writes into the checkout fails by name. (c) Optional, small product change in PR 3: leave `launcher` out of `warn_dirs` when it is in `target_guard._unwatched()`. This takes about 7 git calls × ~24 snapshots on the large checkout out of every design-run test. The launcher-warning path stays covered by `test_design_worktree.py:1064`, which uses a temporary launcher. |
| F2 | `test_live_activity::test_the_index_answers_under_concurrent_builders_and_torn_lines` | 43130726733 pool-followups (passed on re-run); 43305916432 integrity-judge; 43313114699 integrity-judge (passed on re-run); 43543586958 fix/macos-test-suites; 43692760565 structure-infra (passed on re-run) | Re-runs pass. Mode A (4 times) at `:1250`: `assert 110 == 111`. Mode B (1 time, 11-12) at `:1238`: a reader got `[]` | **Most likely a product race.** `live.emit` (`live.py:81-85`) does `fstat` → `pread` of the last byte → `write(separator + line)` with no lock. A concurrent `O_APPEND` writer whose size is already visible before all its bytes land makes the check see a byte that is not a newline, so it adds a spurious `\n`. The journal then has 111 newline-terminated lines while the index counts 110 records. | Fix the product, not the test: take `fcntl.flock(fd, LOCK_EX)` around the fstat/pread/write in `emit` (the same lock the index builder's append should use). Then confirm with a loop of 500 runs on the lab. Mode B (concurrent index builders) needs the same lock on index building; check after the emit fix. Do not quarantine: this test found a real bug. |
| F3 | `test_live_activity::test_page_cost_does_not_grow_with_the_journal` (`:947-948`) | 43612507168 split-evidence (passed on re-run); 43682683223 structure-infra (passed on re-run) | Re-runs pass. The numbers are noise: `{500: 5.85, 5000: 8.66, 20000: 2.62}` ms | It compares wall-clock medians in milliseconds across journal sizes under 4-worker load. It also builds a 270,000-record journal, costing 10–18 s per run. | Measure work instead of time. Count the bytes and records read through `live` (monkeypatch `live.os.pread`/`read` and file opens, or add a `live._stats` counter) and assert that a page at N=270k reads no more than a page at N=500 plus a constant. Counting is exact, so the sizes can shrink to (500, 20,000), saving about 10 s. Fallback: move only the timing assertions to the nightly job. |
| F4 | `test_evaluate_many::test_results_order_mutation` (`:512`) | 43502677784 fix/setup-hook-isolation (passed on re-run); 43620600183 split-evidence | Re-run passed. The margin went from 0.4 s to 2.0 s in #655 (11-14), with no failure since, but the order still depends on sleeps | Its order depends on `time.sleep` | Use the event pattern `test_on_result_fires_in_completion_order` (`:615`) already uses: the slow candidate waits on the fast one's `on_result` Event. This also removes the 2 s sleep. |
| F5 | `test_evaluate_many::test_on_result_fires_in_completion_order` | 43465534872 attempts 1 and 2 (55cb4746) | Its order depended on sleeps | — | **Already fixed** in #639 (rewritten to wait on a `threading.Event`). No action. |
| F6 | `test_concurrent_rounds::test_concurrent_rounds_score_and_decide_as_one_at_a_time`, `..._seeds_first_round_...`, `test_a_budget_stopped_evaluation_leaves_the_candidate_for_a_resume[val/train]` | 43142814178 optimizer-structure-feedback 249bb174 | `a later evaluation of the wave never finished` and `BrokenBarrierError`; not seen since 11-12 (about 150 runs) | `_slow_and_val_by_level` (`:118-155`) makes each wave member wait for the later members while holding a pool thread. If a wave is larger than the evaluation pool, it deadlocks until the 60 s timeout. | Watch. Hardening: have the helper assert `len(wave) <= max_concurrent_evaluations` so a too-large wave fails immediately with a clear message. (Its earlier failures on conc-eval, 42832283243 and 42837771953, were the introducing pull request: real.) |
| F7 | `test_livemath_blocks::test_failed_fork_branch_spine[3]` (`:951`) | 43605791275 patch-first-edits 9c65642d | Happened once, at parallelism 3 only | The fake worker fails **exactly once** (an `O_EXCL` marker), yet **two** items recorded `solve_a: worker failed`. Most likely one item's error leaks into another item's record in the spine's parallel data pass: a **product race**. | Do not quarantine. Reproduce on the lab (`-k "failed_fork_branch_spine and 3"` 200 times) and fix the spine's per-item error bookkeeping. This goes in the tracker, not in these hand-made pull requests. |
| F8 | `test_graph_control_owner::test_a_rerun_behind_a_pause_holds_at_the_reopened_node_until_resumed` (`:341`) | 42713919150 **main** de050e27 | Happened once; the receipt status was `failed` | The receipt's reason was cut off in the assertion message | Make the assertion print the receipt's error/reason, then watch. |
| F9 | `test_network_wait::test_the_lab_outage_is_waited_out_and_the_call_answers` | 43558489753 memory-tried-and-objective | Its sleep recorder caught 1,546 sleeps from another test's leftover thread | — | **Already fixed** in #652 (the recorder now filters by caller frame). No action. |
| F10 | `test_trace_files::test_the_writer_changes_no_request_journal_or_report` (`:183`) | 43690868211 run-a24f4d71100d (the test is now on main) | `wall clock 0.95s` vs `<n>s` | The timings are normalised with `\d+\.\d{3,}`. A time that prints with fewer than 3 decimals (0.950 printed as 0.95) slips through. This is deterministic given the time value. | Normalise the summary line explicitly (`wall clock [\d.]+s` → `wall clock <n>s`, and all `\d+(\.\d+)?s` there), or freeze the clock the report reads. |
| F11 | `test_robustness::test_design_fast_base_double_crash_continues_with_note` (`:1346`) | 43542635531 fix/early-check-probe | ``uv sync --all-groups` killed by SIGABRT: exit 134`` | The runner environment (uv aborted while preparing a worktree environment under load) | Watch. If it recurs, have the test use the existing pre-synced environment or stub `uv sync`. |

### 2b. Real failures (16 attempts: correct reds)

| kind | runs | how to catch it earlier |
|---|---|---|
| stale generated docs (`gen-tutorial`, `documentation.json`/`generated-help.ts`) | 42918796174, 43122544549, 43128529407, 43130545009, 43377148068, 43457105817, 43498183410 | They already fail about 3 min in, during install.sh. Add `pnpm run docs:check` to the lefthook `pre-push` hook (seconds). |
| `test_boundary::test_all_modules_classified` (new module not classified) | 43487980615, 43488594123, 43546305286 (plus 43678180497) | They fail only after the 25-min suite. Add a "fast checks" step before the suite, and run the same files in lefthook pre-push: `test_boundary.py test_registry.py` and `test_cli.py::test_the_command_reference_matches_the_command_line` (under 1 min). |
| the pull request's own bugs | 43384647382 (walker_resume `FakeModel`, 23 tests), 42832283243 and 42837771953 (conc-eval), 43678180497, 43681152758, 43685608883 (code-graph design run), 43488594123 macOS `open.spec.ts` (the pull request that added the macOS job) | — |

## 3. Speed

### Where the time goes (median over 3 green runs; summed test time 114 min)

| file | s | tests | why it is slow |
|---|---|---|---|
| test_delivery | 1083 | 246 | ~90 tests each run the **whole design graph** (`_design`, `_design`+`deliver_design`, `_limited`) at about a 10 s floor on CI. One test, measured locally, started **1,203 git processes** (770 are target-guard snapshots: `rev-parse` ×2, `is-inside-work-tree`, `check-ignore`, `show-toplevel`, `status`, `symbolic-ref`, i.e. ~7 per snapshot × 82 snapshots, plus ~170 on the real checkout through `warn_also`). 5.8 s locally vs 23 s on CI. |
| test_design_worktree | 470 | 152 | whole design runs |
| test_first_self_check / test_self_check | 287 / 195 | 55 / 71 | whole design runs. The 7-case markdown parser test alone is 72 s. |
| test_search_combo | 265 | 87 | outer-loop runs across 9 strategies (several 9-way families at 25–30 s each) |
| test_design_scale | 199 | 95 | whole design runs |
| test_validation_selection | 175 | 84 | 21-case replay of recorded runs (81 s) |
| test_loop_as_mode | 159 | 23 | 5 identical default runs |
| test_runtime_equivalence | 153 | 23 | module fixture: 5 design runs (109 s, charged to the first test) |
| test_executor | 138 | 85 | `test_step_cap_scaling_exceeds_unscaled_cap` **102 s** (1,001 items run twice) |

The top 12 files are 50% of the time and the top 30 are 70%. 431 tests take over 5 s and add up to 74 min. The slowest single tests are runtime_equivalence[walker] (109 s, fixture), the executor step-cap test (102 s), the self_check worst case (44 s), the livemath spine knobs test (31 s), and code_nodes/loop_as_mode/delivery conflict tests (21–28 s each).

**Why it is slow:** process launches (git above all, plus Python workers and Node spine runs) on a 4-vCPU runner with 4 workers competing, real whole-graph runs where a unit call would do, and oversized inputs. There are no real long sleeps; the only one found is 2 s in F4.

### Plan

1. **Shard the Python suite across parallel jobs.** Use a `matrix: shard: [1, 2, 3, 4]`, each job running `-n 4`. Standard runners are free for a public repository, so this costs no money. Add a small hook in conftest (`--shard=i/N`, about 40 lines) that bin-packs **test files** by a committed `tests/ci_durations.json` (file → seconds). Files missing from the JSON are placed by hash. A script, `scripts/ci-durations.py`, refreshes the JSON from the shards' JUnit artifacts. Sharding by file keeps loadscope's guarantees and builds each module fixture once.
   - **This requires splitting `test_delivery.py`** (18 min on one worker) into about 4 files by theme (draft/refine, conflict/reloop, new project/publish, no-pr/kept-local). Each golden module fixture stays with the file that uses it. Also split `test_design_worktree.py` into 2. These are pure moves with no test changes.
   - The TypeScript suite, web acceptance and harness_check run in shard 1 only (about 1 min). Docs regen/check still runs on **every** shard, through install.sh.
   - The expected time is set by the larger of summed time / 16 workers (114/16 ≈ 7.1 min) and the biggest single file after the split (≈4–5 min), plus setup (≈1.5 min with fetch-depth 0). That gives **≈ 9–10 min per shard**, or ≈ 13 min on a slow runner. The macOS job (4–8 min) runs alongside.
   - Alternative (not chosen first): `--dist loadgroup` with `xdist_group` marks on test_create, test_edit_design, test_runtime_equivalence and the golden fixtures. It balances better but changes scheduling for all ~250 files. Try it later behind the conftest checkout check from F1(b).
2. **A fast-checks step** before the shards (or as shard 0): docs:check, then `pytest -q -x test_boundary.py test_registry.py test_cli.py::test_the_command_reference_matches_the_command_line` (under 1 min). Combined with the matching lefthook pre-push jobs, this catches 11 of the 16 real reds before push or within 3 min.
3. **Add pytest-timeout** (dev group) with `--timeout=900` in CI. The existing `@pytest.mark.timeout` marks start working, and a hang fails one test instead of eating 60 min. Change **`-rs` to `-rfEs`** so failures are listed at the end of the log.
4. **Trim and condense (PR 3, below).** About 7 min of summed time saved without the delivery migration, about 20–25 min with it.
5. **Fewer process launches (follow-up, product code, so a design run).** A target-guard snapshot makes about 7 git calls. One `git status --porcelain=v2 --branch -z` (which reports HEAD, branch and changes) plus the cached tree lookup would make it 1–2. That is roughly 60% fewer git launches in every design-run test, and it also helps the Mac launch budget the macOS job guards. It changes guard behaviour, so it should go through design mode, not a hand-made pull request.
6. **Nightly workflow (adds coverage, moves nothing):** `--run-structure-fuzz-long` (today it runs only by hand, `docs/outer-loop.md:1867`) and optionally Lean core so the 4 `test_math_pack` Lean tests stop being permanent skips. **Every test that runs on push today keeps running on push, including all mutation checks** (~75 s in total; list in §4).

Expected CI time: **26–37 min today → ≈10–12 min after PR 2 → ≈8–10 min after PR 3** (macOS job in parallel; red runs drop by about 60% after PR 1).

## 4. Trim (conservative; each row names what else covers the behaviour)

| test | CI s | change | still covered by | saving | confidence |
|---|---|---|---|---|---|
| `test_executor.py:1536` `test_step_cap_scaling_exceeds_unscaled_cap` | 102 | Set `default_max = 20`, pass `max_steps=20` to both `execute` and `_NoScaleExecutor`, use 11 items | Same test, same two halves (scaled succeeds, unscaled fails) | ~100 s | high |
| `test_executor.py:1247` `test_large_benchmark_completes_under_step_cap` | 26 | 30 items with `max_steps=20` (or merge into `:1536`) | `:1536` scaled half | ~24 s | high |
| `test_first_self_check` `test_markdown_around_the_updated_tests_line_is_not_part_of_the_test_id[*]` (7 cases) | 72 | Make the 7 cases direct calls to `_plain` / `UPDATED_TESTS_LINE` (`design_ops.py:3644`); keep 1 end-to-end case | `test_the_reviewers_and_the_qa_judge_see_the_tests_the_builder_says_it_updated` (end-to-end wiring) | ~65 s | high |
| `test_self_check.py:1467` `test_the_worst_case_at_the_knobs_bounds_still_ends_in_the_stop` | 44 | Replace the run with a static assertion of the worst-case reloops vs `max_iterations` over the knob bounds | `:312` `test_the_worst_case_ends_in_the_build_loop_stop_not_the_iteration_limit` (the same behaviour as a real run) | ~44 s | med |
| `test_design_worktree.py:1053` `test_a_second_run_on_a_project_names_the_earlier_runs_branches` | 18 | Merge its warning assertion into `:147` (which makes the same two runs) | `:147` `test_the_same_idea_again_is_a_new_branch_of_the_same_project_from_main` | ~17 s | high |
| `test_livemath_skill_graph.py:500` `test_spine_matches_walker[1]` | 16 | Add the score and record assertions to `:515`; keep `[3]` | `:515` `test_spine_carries_the_system_prompt_and_task_knobs` (same graph, parallelism 1, same answers) | ~16 s | med |
| `test_delivery.py:4580` `test_conflict_send_back_stays_bounded_with_multi_megabyte_files` | 25 | 3,000 lines instead of 70,000 (every bound is still crossed if the bounding code is removed); rename "large files" | same test | ~10 s | med |
| `test_delivery.py:5737` `test_unresolved_conflict_stops_after_budget` | 27 | **Fix: it asserts almost nothing.** Add `assert not result2.exit_ok` and a reloop-count check. Use `_move_main_then_deliver` instead of a whole first run | `test_merge_failure_leaves_worktree_clean` (`:5175`) covers only the clean worktree | ~10 s plus a real check | high |

**Kept:** all tests that are always skipped on CI are documented manual or local gates and none is obsolete: the integration tests (`docs/verification.md:13,252`), the Lean/mathlib tests (`--toolchain-gate`), the fuzz long tier, `test_git_exe.py:139` (macOS only), `test_parity_gap` (a placeholder until a gap is listed) and `test_sealed_suite.py:76` (run by its sibling in a child session). So there are no deletions.

**Mutation checks (must keep running on every push):** the `test_structure_fuzz` `test_mutation_*` set (16 tests, ~46 s; 23.8 s of it is the walker data-pass check), the `test_tool_policy` `test_mutation_remove_*` tests (5), the `test_evaluate_many` `*_mutation` tests (3), `test_budget_seams::test_rollout_counting_mutation`, and the docstring-level mutation checks in test_nn, test_search_combo, test_code_nodes, test_skillopt, test_adoption, test_dsh_profile and test_set_knob_patch. About 75 s in total. Nothing in this plan moves them.

## 5. Condense (same coverage, fewer runs or lines)

| area | fold | saving | what stays covered | risk |
|---|---|---|---|---|
| `test_delivery.py`: ~90 whole-graph tests | Move tests whose assertions read only the delivery record, pull requests or gh calls onto the existing `_new_golden` / `_opened_golden` fixtures (`:2338-2357`) plus a direct `delivery.deliver_design(...)` call. 26 tests already do this at under 1 s each. Keep whole-graph runs for routing tests (limit stops, conflict reloops). | 10–15 min summed if about 60 move | Same assertions on the same delivery code | med: each move must confirm the golden run's facts match what the test's own run produced |
| `test_loop_as_mode.py` | One module-scoped `default_run` fixture for the 5 default-settings runs (`:110, :119, :153, :240, :254`) | ~45 s | All assertions, on one run | low |
| `test_design_worktree` refusal family | The 6-case `test_an_emission_materialise_cannot_apply_goes_back_to_emit_with_the_reason` (55 s): test each refusal reason directly, keep 2 end-to-end cases (one collect, one materialise) | ~35 s | Each reason's text, plus the send-back wiring end to end | med |
| Duplicated brief tests | `test_design_worktree.py:1898` + `:2060` (bare name vs `path::id`) become one parametrized test, or a direct matcher test for one form | lines; ~11 s with the direct test | Both forms | med |
| Parity helpers | `_spine(` is defined in 10 files and `_walk(` in 8. Fold them into one helper in `tests_support` | lines only | Same | low |
| Duplicated helpers | `_git_init` ×4 vs `tests_support.git_repo`; `class FakeModel` ×5; `_commit` ×5; identical `_mode`/`_design` copies. Move the identical ones into `tests_support`; keep the variants that script graph-specific answers | lines only | Same | low–med |
| CI steps | Nothing builds twice inside a job. With sharding, the TypeScript suite and acceptance run once (shard 1), not per shard | ~1 min CPU per shard | Same | low |

## 6. Pull-request breakdown (smallest risk first; all hand-made)

| PR | contents | risk | effect |
|---|---|---|---|
| **1. Flake fixes** | F1(a) static op modules for test_data_parallel/test_evaluate_many; F1(b) conftest check that the checkout is unchanged; F2 `flock` in `live.emit` (3 lines of product code, tell the user it is product code); F3 page_cost counts bytes read instead of timing, and its sizes shrink; F4 results_order_mutation waits on events; F10 trace_files normalises the summary line; F8 clearer receipt assertion; `-rs` → `-rfEs` | low (tests only, plus one lock) | Removes the causes behind 16 of the 23 flaky attempts (F1, F2, F3, F4, F10). F5 and F9 are already fixed. F6, F7 (the spine race), F8 and F11 are to watch or investigate |
| **2. CI shape** | conftest `--shard` hook + `tests/ci_durations.json` + refresh script; split `test_delivery.py` (4 files) and `test_design_worktree.py` (2 files) as pure moves; workflow: fast-checks step, 4-shard matrix, TypeScript/acceptance in shard 1, per-shard JUnit upload, pytest-timeout `--timeout=900`; lefthook pre-push gets `docs:check` and the boundary/registry/command-reference tests; nightly workflow for the fuzz long tier | med (workflow + moves; every test still runs, and the shard hook can be checked by comparing collected counts with a full run) | ~26–37 min → ~10–12 min. 11 of the 16 real reds caught before push or within 3 min |
| **3. Trim + condense** | §4 rows (executor ×2, first_self_check parser, self_check worst case, design_worktree merge, livemath `[1]`, delivery conflict shrink + vacuous-test fix); loop_as_mode shared fixture; refusal family; helper consolidation; F1(c) skip the unwatched launcher in `warn_dirs` | low–med (each row names what covers it; mutation checks untouched) | About 7 min of summed time, about 2 min of wall time; one real test defect fixed |
| Follow-ups (not hand-made pull requests) | F7 spine per-item error race (tracker + repro); target-guard snapshot batching (§3.5, through design mode); test_delivery golden-fixture migration (§5, 10–15 min summed); a `loadgroup` trial | — | — |

## Evidence files (scratchpad)

- `runs.json`: the 200-run list. `logs/<run>-<attempt>.log`: failed-attempt logs. `fails.json`: parsed failures per attempt.
- `art/<run>/python-suite.xml`: per-test CI times.
- Local measurement: `test_delivery::test_a_persons_own_draft_is_not_marked_ready` took 5.8 s locally (2.9 setup + 2.9 call) vs 23.0 s on CI, with 1,203 git launches, recorded with an audit-hook `sitecustomize` (on the older checkout ~/work/refactory @1ea2134e43).
