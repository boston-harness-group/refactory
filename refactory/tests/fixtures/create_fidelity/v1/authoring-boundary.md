## Benchmark verification ownership (authoring context v2)

The independent 56-case checker is deliberately unavailable inside every
authoring lane. The operator runs it against the unchanged delivered CLI after
Create finishes, outside the authoring sandbox. This follows the benchmark's
existing isolation protocol; it does not remove or reduce any B01–B10 requirement.

During Create, implement and independently review the complete public contract,
run local native and CLI tests, and make every normal build, QA, documentation
and completion decision from actual available evidence. Keep the operator's
56-case verification explicitly pending in the strategy, configuration and
reports. Do not claim it passed, recreate it as an allegedly independent checker,
or make its deliberately hidden file an internal authoring prerequisite.
Report any local defect or missing local test honestly through the normal gates.

Completing Create only delivers a candidate for external acceptance. Benchmark
acceptance still requires the actual frozen checker to execute exactly 56 cases,
native reuse and error checks, and the separate authoring-fidelity scorecard.
An absent, failed or zero-case checker result cannot pass benchmark acceptance.
