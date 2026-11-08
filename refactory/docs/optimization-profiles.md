# Optimization profiles

A profile is a named, immutable version of search settings and the user's objective. A run writes its exact resolved settings, source identity, input fingerprints and validation contract to `resolved-profile.json`; changing the reusable profile later does not edit that run.

```sh
refactory profile-save --name small-search --spec '{"settings":{"optimizer":"random","edit_preset":"knob_only","epochs":1,"batch_size":2,"max_evaluations":2,"repeats":2,"max_elapsed_s":30,"validation_reserve_s":3},"contract":{"objective":"Compare a bounded pair of candidates","required_artifacts":[]}}'
refactory profiles
refactory profile-resolve circle_packing/adaptive --profile small-search --profile-version 1
refactory optimize circle_packing/adaptive --profile small-search --profile-version 1 --runs-root .factory/profile-demo
refactory comparison --run .factory/profile-demo
refactory adopt --run .factory/profile-demo --as profile_winner
refactory package smoke profile_winner/adaptive
```

Saving the same name creates a new version. A spec holds only `settings` and `contract`; an unknown field or setting, a value of the wrong type (a string seed, `true` as a repeat count) or a strategy option the strategy does not read is refused, never ignored. CLI overrides are explicit: the snapshot's `settings` hold the merged values and its `overrides` what the command line gave. A setting the command line leaves out keeps the profile's value, including `parts_manifest` and `strategy_options`; `--knob` and `--strategy-option` add to the profile's knobs and strategy options. Profiles support the existing strategy, optimizer and evaluator; starting/frozen knobs, edit preset and a narrower allowed-operation list; epochs, batch size, evaluation cap, retained parents, proposal attempts, seed, holdout and parsimony; repeats, shared resources and worker/model routing. `refactory optimize --help` lists every option.

In the native Refactory panel, select a scored mode and expand **Optimization profiles and full search controls**. Select a saved version or edit controls and save a new version. The mode's existing knob fields choose starting values. **Preview launch contract** uses the same resolver as the CLI and launch service. Save objective or additional-artifact changes before launching. Existing live notes, candidate keep/reject, pause/resume/stop, explicit adoption and using the adopted mode once remain available.

**Candidate changes, measurements and validation** includes successful, failed and unmeasured states. Expand a candidate to inspect its lineage, graph/prompt/knob diff, raw and adjusted score, real trial count/variation, required-artifact outcome and duration. Missing token/cost telemetry is unavailable, never zero. Held-out winner validation is separate from training and never enters proposal context. The mode still owns Task definitions, scoring code and invariants; a profile cannot redefine them.

By default `repeats=1`. Opt in to 2–10 independent measurements per candidate and winner audit. Trials consume shared resources, including final validation. One failed or incomplete trial makes that candidate ineligible; its partial measurements remain visible. Ranking maximizes the mean validated score minus `parsimony × node count`. Cost and latency are descriptive rather than hidden selection advantages. A larger observed mean is not statistical proof; finding no improvement is a valid result.

Runs that share a seed can skip re-measuring it. `python -m refactory_dsh.seed_cache export RUN_DIR seed-results.json` writes the seed's first result on each part of a run, and `--seed-results seed-results.json` makes a later run read the seed's first evaluation of each part from that file instead of running it. With `repeats` above 1 every trial is exported and reused. A cached result is used only for the same seed, on the same item set, under the same score context: the mode, its implementation and inputs, the task-model route, the evaluator, the item timeout, the model client's inactivity limit and the mode package's files. Search settings and the proposer's route are not part of it. A run whose context differs stops with the differing fields named, and so does a run whose seed is another candidate or whose part holds other items. A train minibatch other than the cached one is measured for real and journaled as `seed_cache_miss`. A reused measurement is journaled as `seed_cache_hit`, marked `cached_from` in the candidate's measurements, and listed under `seed_results` in `report.json` and in `report.md`. Each part is served once per run, and cached results don't count toward `max_rollouts`. Every run writes its score context to `score-context.json`.

Adoption records the exact winning state, profile snapshot and held-out audit in its provenance. Historical configurations without profiles remain readable. Profiles apply to scored modes, including generated Create/Edit packages; standalone Design artifacts have no separate optimizer.

See [shared budgets and fast modes](budgets-fast-modes.md), [model and worker routing](models.md), and [live controls and recovery](live-control.md) for the resource, execution and continuation contracts used by these settings.
