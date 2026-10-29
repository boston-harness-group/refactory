category: SPEC_CHANGE_REQUESTED
workspace:
  source_root: /Users/rawhad/3_Resources/external_libs/refactory-dsh
  worktree_root: /Users/rawhad/3_Resources/external_libs/refactory-dsh/.hai-worktrees/INT-001-activity-verbosity
  branch: hai/INT-001-activity-verbosity
  base_commit: 02a86790a6acc263d77d0f6588d62bcbfda6a055
session_id: session-a0e27736-9599-4b1e-9464-a8adf017b3d8
verdict: FAIL
milestone: M1 (console contract) - build-loop finding, raised during implementation
spec_path: docs/refactory-activity-verbosity-spec.md @ 5f4475e289eccbf092d0618dbad2cd0673fd167f4806d44d6f1d299abec346a0 (revision 4)
affected_ids: [R-1.4, R-1.6.1, T-24, T-25, gate 7 (test_bounded_activity_page)]

blocked_done_gate: >
  §12 gate 7, `test_bounded_activity_page`, gap case sentence: "a 401st line that
  is a malformed *complete* record makes total == 401, serves activity.events[-1]
  with kind == 'journal_gap', keeps has_more == (cursor < total) true at
  after == 400".

build_evidence:
  - docs/refactory-activity-verbosity-spec.md:1442-1447 (the disputed sentence)
  - docs/refactory-activity-verbosity-spec.md:1440 (the identity it must not contradict)
  - refactory/refactory_dsh/live.py - events() counts a malformed complete line as
    a journal_gap record (the counted-record rule R-1.3 depends on)
  - M1 builder report: at after == 400 the page serves the gap itself, so
    cursor == 401 == total and has_more is false, not true; the identity
    has_more == (cursor < total) holds exactly in every case.

requested_contract_delta: >
  Narrow, one-word class of correction in gate 7 only: the gap case SHALL assert
  has_more == (cursor < total) on every answer (which at after == 400 with
  total == 401 means has_more == false, because the page serves the gap and
  advances cursor to total), and SHALL assert the true reachability fact instead
  of "true at after == 400" - that from after == 0 the gap record is reachable on
  the third page (200-record pages: records 0-199, 200-399, 400). No requirement
  in §2.2 (R-1.4, R-1.6.1) changes; the identity they state is already correct and
  is what the implementation satisfies.

reason_not_code_only: >
  The implementation is correct and matches R-1.4's identity; only the gate's
  expected value for one assertion is wrong. Fixing code to make has_more true at
  after == 400 would require serving fewer than page_size records or reporting a
  cursor below total, which would violate R-1.4 itself. The contract text has to
  change, so this is a spec amendment rather than a code fix.

hai_alignment: >
  INT-001, DEC-001, DEC-002 unaffected - no change to human-approved state is
  required. The amendment only makes an executable gate agree with the normative
  identity that already traces to INT-001/DEC-001.

implementation_impact: >
  refactory/tests/test_live_activity.py::test_bounded_activity_page gap
  assertions (assert the identity unconditionally; assert third-page reachability
  from after == 0). M1 gate 7, and M4's full gate run. No production code change,
  no other test, no member or option shape is affected.
