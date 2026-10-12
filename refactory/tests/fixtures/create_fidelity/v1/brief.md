Create a reusable native mode named `booking_conflicts_bench` that accepts the runtime parameter `bookings_path`, reads that JSON file and writes `conflicts.json` into that invocation's run directory. A model may author the mode; executing the generated booking computation must not require a model. Package a Python 3.11+ standard-library CLI named `booking_conflicts.py` and a README alongside the framework's native mode artifacts.

| ID | Required behavior |
|---|---|
| B01 | Input is a JSON array of bookings; an empty array is valid. |
| B02 | Each booking has nonempty string `id` and `room`, and integer `start`/`end` minute offsets with `0 <= start < end <= 1440`. Booleans are not integers for this contract; extra fields may be ignored. |
| B03 | Booking IDs are globally unique across rooms. |
| B04 | Conflicts require exactly equal room strings and half-open interval overlap: `max(start1,start2) < min(end1,end2)`. Touching endpoints do not conflict. |
| B05 | Output is exactly an object with `conflicts`: unique ID pairs, IDs sorted within pairs and pairs sorted lexicographically. Input permutation does not change parsed output. |
| B06 | CLI is `python booking_conflicts.py --input INPUT.json --output OUTPUT.json`, works from an unrelated working directory with absolute paths, and returns 0 on valid input. |
| B07 | Invalid input returns nonzero with useful stderr, creates no output when absent, and leaves an existing CLI output byte-for-byte unchanged. A native mode invocation must propagate failure honestly without claiming a valid result. |
| B08 | Mode is discoverable, inspectable, validated, documented and runnable through the framework's actual native entry points. Framework-specific artifact formats are allowed. |
| B09 | Invoke the same generated package twice on distinct valid input files in separate run directories, without package modifications. Observe distinct expected results and correctly located `conflicts.json` files. |
| B10 | README documents installation/discovery and a native invocation with runtime `bookings_path`, plus the standalone CLI and ordinary examples. |

Both authoring lanes receive this content unchanged, plus only their framework-specific invocation instructions. Generated work stays local to disposable benchmark projects; no benchmark child may publish GitHub artifacts or modify framework source. Engine implementation and its eventual PR are separate from generated test projects.

Example native executions: input A is `a/A/[0,10)` and `b/A/[5,15)`, expecting `[["a","b"]]`; input B is `c/A/[0,10)` and `d/A/[10,20)`, expecting `[]`. A third invocation with duplicate IDs must fail. The full independent CLI checker covers 56 cases, including invalid-input preservation. Freeze its bytes and assert the expected case count; do not treat 0 tests as success.
