# Milestone plan: INT-001 activity verbosity

Derived from the spec's §13 and its done gates (§12). Milestones are ordered by
dependency: the console contract first, then the DSH boundary and service, then
the panel that consumes both, then the integrated gate run.

| Milestone | Outcome | Spec done-gates | Spec tests | Non-goals deferred |
|---|---|---|---|---|
| M1 console contract | A paged, cost-bounded activity read and a journal that no longer grows quadratically | 1, 2, 3, 7, 8 | T-1…T-10, T-24 | boundary verdict (M2), panel levels (M3) |
| M2 boundary + service | Truncation is its own failure with facts the seam delivers; service forwards both cursors | 4, 5, 6 | T-11…T-15, T-23 | panel rendering (M3) |
| M3 panel | Three levels with standard as the default, full history reachable at every level | 4, 5 | T-16…T-22 | integrated use cases (M4) |
| M4 integration | Every §12 gate green together, §11 manual checklist executed by the reviewer | 1–8 | all | none |

Write scopes per milestone (advisory): M1 `refactory/refactory_dsh/`,
`refactory/tests/`; M2 `packages/refactory/`; M3 `packages/client/ui-refactory/`;
M4 `logs/`, `docs/refactory-activity-verbosity-spec.md` (revision log only if a
`SPEC_CHANGE_REQUESTED` is approved).
