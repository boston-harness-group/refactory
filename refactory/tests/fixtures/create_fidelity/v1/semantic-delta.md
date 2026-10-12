# Reviewed reference delta

Reference: `3ab6f8f7ca792c874f2654574d7f7fd6dd18b896`; historical comparison:
`3a74443e2b53dd55e847cea6d7af4b8d4865e279`. The graph was reconstructed from
an isolated git archive, yielding 29 nodes and 33 edges. Topology is unchanged.

* `health_checker` and `code_reviewer` now have explicit default review prompts.
* Strategists must specify an acceptance outcome, scope, and evidence for each
  hypothesis. Builders must deliver that evidence at or above the stated scope.
  Code review adds an eighth, mandatory acceptance-evidence category.
* Health checking adds an advisory patch-target audit. Findings do not alone
  change the gate verdict.
* Builder PR reuse/preservation instructions changed. Benchmark runs prohibit
  publishing generated projects, so this is an explicit artifact-only override
  in both lanes, not permission to publish test artifacts.
* Shared primitives validate function nodes and change DataNode topology and
  runtime inputs. Create has no DataNode; the native adapters must nevertheless
  validate generated modes against their own pinned runtime.

The graph fixture includes complete inherited role prompts, not only node text.
Known reference executor defects (including malformed bootstrap shell quoting)
are failures to record; they are not behavioral requirements to reproduce.
JSON director plans and prose quality summaries are separate artifacts in DSH.

Future-run native bootstrap template mapping supplies the exact canonical
`templates/factory_config.md` from the same pinned source commit as a production
resource, independent of this test oracle. Its SHA-256 is
`50d8a754f707e915a0b13bd8a235d201b1930939d3b6801958dde9242e8a672e`.
The existing `init_user_intent` node publishes an unchanged run-local template
and source/copy provenance under `.factory/bootstrap/`. `create_factory_md`
declares both as inputs, reads/copies/fills the template, and delivers the
filled configuration through normal input/output-bound agent receipts. The
input provenance proves which canonical bytes were supplied; it does not
self-certify that a model copied/filled them or that acceptance passed.

The specific source assignment to create `factory.md` and that stage's declared
output takes precedence over the inherited CEO's general config/non-review
write prohibition only for this configuration stage. It does not authorize
application-code edits, arbitrary config changes or another specialist's work.
The six source-required fields remain concrete. Canonical `Eval`/`Command` and
`Threshold` adapt to actual native validation and requested acceptance commands
for intentionally unscored artifact modes; no default `0.8` optimizer score,
`eval/score.py` or Task is fabricated. Requested scored contracts are preserved.
Remote Target Branch/publication defaults become explicitly not applicable in
this local artifact-only run. Eval Spec/research sections activate only from
actual source context; absent sections may keep harmless commented examples or
be explicitly marked not applicable. Smoke outputs use fresh project/private
TMPDIR paths that respect the invocation's readable/writable boundary.

This adds only candidate artifact declarations to existing nodes, retaining
29 nodes/33 edges and original source prompts. It changes future candidate graph
identity; saved checkpoints and frozen v6 artifacts are not rewritten. The v6
operator review's missing literal template-copy provenance remains historical
unavailable evidence, not retroactively repaired by this mapping.

No user patch is transplanted. The local patch inventory in `baseline.json`
records digests and disposition. Public Create remains the rollback baseline.

An additional optional IR-compatibility run against this exact remote snapshot
rejects DSH's existing `command_timeout_s` and `resource_role` serialization
fields. The same failure reproduces on the frozen `ec99b4ec` DSH baseline for
both counting modes. This is a pre-existing cross-library schema mismatch,
separate from the independent Create behavior contract; the compatibility path
is not claimed as verified. The installed-environment regression suite skips
that optional library unless it is explicitly provided.
