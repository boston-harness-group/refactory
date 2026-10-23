// Generated offline by refactory_dsh.tools.gen_discovery. Do not edit.
export const CHAT = {
  "adopt": {
    "arguments": {
      "as": {
        "default": null,
        "description": "the new package name",
        "flags": [
          "--as"
        ],
        "nargs": null,
        "positional": false,
        "required": true,
        "type": "string"
      },
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "as": "as_name",
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot",
      "as"
    ],
    "purpose": "Publish the exact measured winner as a new package without changing the source; overfit adoption needs the explicit CLI override.",
    "required": [
      "run",
      "as"
    ],
    "verb": "adopt"
  },
  "artifacts": {
    "arguments": {
      "contents": {
        "default": false,
        "description": "",
        "flags": [
          "--contents"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "contents": "contents",
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot",
      "contents"
    ],
    "purpose": "List recorded run artifacts; optionally include contents for evidence review.",
    "required": [
      "run"
    ],
    "verb": "artifacts"
  },
  "best": {
    "arguments": {
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "state": {
        "default": false,
        "description": "include the full state",
        "flags": [
          "--state"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      }
    },
    "fields": {
      "run": "run",
      "runsRoot": "runs_root",
      "state": "state"
    },
    "order": [
      "run",
      "runsRoot",
      "state"
    ],
    "purpose": "Read the measured winning state before adoption; inspect held-out validation and required artifacts.",
    "required": [
      "run"
    ],
    "verb": "best"
  },
  "candidate": {
    "arguments": {
      "candidate": {
        "default": null,
        "description": "",
        "flags": [],
        "nargs": null,
        "position": 0,
        "positional": true,
        "required": true,
        "type": "string"
      },
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "candidate": "candidate",
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot",
      "candidate"
    ],
    "purpose": "Read one candidate and its parent diff using the run and candidate IDs.",
    "required": [
      "run",
      "candidate"
    ],
    "verb": "candidate"
  },
  "comparison": {
    "arguments": {
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot"
    ],
    "purpose": "Inspect all optimization trials, including failed and unmeasured candidates, plus exact resolved settings.",
    "required": [
      "run"
    ],
    "verb": "comparison"
  },
  "create": {
    "arguments": {
      "approve": {
        "default": [],
        "description": "answer a gate a person owns, as <gate>=<value> (repeatable); gate_strategy is one, so a headless create is told the answer",
        "flags": [
          "--verdict"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "attempts": {
        "default": 3,
        "description": "",
        "flags": [
          "--attempts"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "brief": {
        "default": "",
        "description": "",
        "flags": [
          "--brief"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "dataNode": {
        "default": false,
        "description": "build the mode as a DataNode that iterates its subgraph per item",
        "flags": [
          "--data-node"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "extends": {
        "default": "",
        "description": "",
        "flags": [
          "--extends"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "inputs": {
        "default": [],
        "description": "an input the produced mode must accept, named the way a knob or a file would be (repeatable); validation refuses a mode with no way to receive it",
        "flags": [
          "--input"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "model": {
        "default": "",
        "description": "model id for this authoring run",
        "flags": [
          "--model"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "name": {
        "default": "",
        "description": "",
        "flags": [
          "--name"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "provider": {
        "default": "",
        "description": "provider route for this authoring run",
        "flags": [
          "--provider"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "resume": {
        "default": "",
        "description": "continue a failed run from its directory",
        "flags": [
          "--resume"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "sample": {
        "default": [],
        "description": "a knob value the validation run should use (repeatable), so the smoke exercises the input path rather than the built-in fixture",
        "flags": [
          "--sample"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "singleShot": {
        "default": false,
        "description": "build a harness for one run of one thing, with no Task binding; one of --task or --single is required",
        "flags": [
          "--single"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "task": {
        "default": null,
        "description": "the class of work: a Task the produced harness binds a data node to, so the harness runs its subgraph once per item the Task lists",
        "flags": [
          "--task"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "approve": "verdict",
      "attempts": "attempts",
      "brief": "brief",
      "dataNode": "data_node",
      "extends": "extends",
      "inputs": "input",
      "model": "model",
      "name": "name",
      "provider": "provider",
      "resume": "resume",
      "sample": "sample",
      "singleShot": "single_shot",
      "task": "task_ref"
    },
    "order": [
      "name",
      "brief",
      "extends",
      "attempts",
      "dataNode",
      "task",
      "singleShot",
      "approve",
      "inputs",
      "provider",
      "model",
      "resume",
      "sample"
    ],
    "purpose": "Create a reusable mode package from a brief; declare task or singleShot. For user gates prefer wf_open default_pack/create; headless approvals require explicit authorization.",
    "required": [
      "name",
      "brief"
    ],
    "verb": "create"
  },
  "design": {
    "arguments": {
      "approve": {
        "default": [],
        "description": "answer a gate a person owns, as <gate>=<value> (repeatable)",
        "flags": [
          "--verdict"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "attempts": {
        "default": 3,
        "description": "",
        "flags": [
          "--attempts"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "base": {
        "default": "",
        "description": "the commit, branch or tag the run starts from",
        "flags": [
          "--base"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "brief": {
        "default": "",
        "description": "",
        "flags": [
          "--brief"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "check": {
        "default": "",
        "description": "the command that proves the artifact works",
        "flags": [
          "--check"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "generate": {
        "default": "",
        "description": "a command run in the worktree after dependencies are installed, before setup and each check; whatever it changes in tracked files is amended into the attempt. A generator that itself needs a build includes that build in its own command",
        "flags": [
          "--generate"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "model": {
        "default": "",
        "description": "model id for this authoring run",
        "flags": [
          "--model"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "name": {
        "default": "",
        "description": "",
        "flags": [
          "--name"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "noPr": {
        "default": false,
        "description": "keep the work on the run's local branch instead of opening a pull request",
        "flags": [
          "--no-pr"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "noWorktree": {
        "default": false,
        "description": "write the artifact under .factory/artifacts instead of a worktree",
        "flags": [
          "--no-worktree"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "provider": {
        "default": "",
        "description": "provider route for this authoring run",
        "flags": [
          "--provider"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "refine": {
        "default": "",
        "description": "an open pull request to follow up (its URL or owner/repo#number); the run starts from its branch and updates it",
        "flags": [
          "--refine"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "resume": {
        "default": "",
        "description": "continue a failed run from its directory",
        "flags": [
          "--resume"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "reviewComments": {
        "default": false,
        "description": "with --refine: add the pull request's reviews and comments to the steering",
        "flags": [
          "--review-comments"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "setup": {
        "default": "",
        "description": "a command run in the worktree before each check",
        "flags": [
          "--setup"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "target": {
        "default": "",
        "description": "the git repository to change, in a worktree of its own (default: a new project)",
        "flags": [
          "--target"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "approve": "verdict",
      "attempts": "attempts",
      "base": "base",
      "brief": "brief",
      "check": "check",
      "generate": "generate",
      "model": "model",
      "name": "name",
      "noPr": "no_pr",
      "noWorktree": "no_worktree",
      "provider": "provider",
      "refine": "refine",
      "resume": "resume",
      "reviewComments": "review_comments",
      "setup": "setup",
      "target": "target"
    },
    "order": [
      "name",
      "brief",
      "check",
      "target",
      "base",
      "setup",
      "generate",
      "noWorktree",
      "noPr",
      "refine",
      "reviewComments",
      "attempts",
      "approve",
      "provider",
      "model",
      "resume"
    ],
    "purpose": "Build and check software from a brief and a check command, on its own branch in a git worktree: of the repository `target` names, or of a new project. Once its reviews pass it opens a pull request (a new project's private GitHub repository is created then), unless noPr. No reusable mode is required. Prefer wf_open default_pack/design with user gates.",
    "required": [
      "name",
      "brief",
      "check"
    ],
    "verb": "design"
  },
  "docs": {
    "arguments": {
      "limit": {
        "default": 6000,
        "description": "maximum text characters, 1..12000",
        "flags": [
          "--limit"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "offset": {
        "default": 0,
        "description": "character offset within page or section",
        "flags": [
          "--offset"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "page": {
        "default": "",
        "description": "exact page ID from the docs index",
        "flags": [
          "--page"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "query": {
        "default": "",
        "description": "words to search in shipped documentation",
        "flags": [
          "--query"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "section": {
        "default": "",
        "description": "section ID from the docs index",
        "flags": [
          "--section"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "limit": "limit",
      "offset": "offset",
      "page": "page",
      "query": "query",
      "section": "section"
    },
    "order": [
      "page",
      "query",
      "section",
      "offset",
      "limit"
    ],
    "purpose": "Search or read bounded sections of the complete shipped documentation by exact index IDs; omit page/query for the index.",
    "required": [],
    "verb": "docs"
  },
  "edit": {
    "arguments": {
      "allowRegression": {
        "default": false,
        "description": "skip original-mode execution and comparison; still validate the revision",
        "flags": [
          "--allow-regression"
        ],
        "nargs": 0,
        "positional": false,
        "required": false,
        "type": "boolean"
      },
      "approve": {
        "default": [],
        "description": "answer a gate a person owns, as <gate>=<value> (repeatable)",
        "flags": [
          "--verdict"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "attempts": {
        "default": 3,
        "description": "",
        "flags": [
          "--attempts"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "brief": {
        "default": "",
        "description": "",
        "flags": [
          "--brief"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "mode": {
        "default": "",
        "description": "",
        "flags": [
          "--mode"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "model": {
        "default": "",
        "description": "model id for this authoring run",
        "flags": [
          "--model"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "name": {
        "default": "",
        "description": "",
        "flags": [
          "--name"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "provider": {
        "default": "",
        "description": "provider route for this authoring run",
        "flags": [
          "--provider"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "resume": {
        "default": "",
        "description": "continue a failed run from its directory",
        "flags": [
          "--resume"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "sample": {
        "default": [],
        "description": "a knob value the validation run should use (repeatable)",
        "flags": [
          "--sample"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      }
    },
    "fields": {
      "allowRegression": "allow_regression",
      "approve": "verdict",
      "attempts": "attempts",
      "brief": "brief",
      "mode": "mode",
      "model": "model",
      "name": "name",
      "provider": "provider",
      "resume": "resume",
      "sample": "sample"
    },
    "order": [
      "mode",
      "name",
      "brief",
      "allowRegression",
      "attempts",
      "approve",
      "provider",
      "model",
      "resume",
      "sample"
    ],
    "purpose": "Revise a reusable mode package and validate it before QA and final intent audit. allow_regression skips original execution/comparison; revised validation remains mandatory. Prefer wf_open default_pack/edit when a person is present.",
    "required": [
      "mode",
      "name",
      "brief"
    ],
    "verb": "edit"
  },
  "help": {
    "arguments": {
      "topic": {
        "default": "",
        "description": "operation or mode ID; omit for the live index",
        "flags": [
          "--topic"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "topic": "topic"
    },
    "order": [
      "topic"
    ],
    "purpose": "Discover live mode/operation schemas and supported interfaces before constructing a request; topic selects one capability.",
    "required": [],
    "verb": "help"
  },
  "leaderboard": {
    "arguments": {
      "limit": {
        "default": 20,
        "description": "",
        "flags": [
          "--limit"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "limit": "limit",
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot",
      "limit"
    ],
    "purpose": "Compare measured candidate scores and parent diffs; failed/unmeasured candidates are not winners.",
    "required": [
      "run"
    ],
    "verb": "leaderboard"
  },
  "modes": {
    "arguments": {},
    "fields": {},
    "order": [],
    "purpose": "Choose from the actual installed modes, full knob constraints, nodes/roles, ports and user gates. Unscored modes must be run, not optimized.",
    "required": [],
    "verb": "modes"
  },
  "profile-resolve": {
    "arguments": {
      "knobs": {
        "default": null,
        "description": "",
        "flags": [
          "--knob"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "mode": {
        "default": null,
        "description": "",
        "flags": [],
        "nargs": null,
        "position": 0,
        "positional": true,
        "required": true,
        "type": "string"
      }
    },
    "fields": {
      "knobs": "knob",
      "mode": "mode"
    },
    "order": [
      "mode",
      "knobs"
    ],
    "purpose": "Preview a scored mode\u2019s effective optimization settings, immutable contract and worker provenance without launching a search.",
    "required": [
      "mode"
    ],
    "settings": true,
    "verb": "profile-resolve"
  },
  "profiles": {
    "arguments": {},
    "fields": {},
    "order": [],
    "purpose": "List immutable saved optimization profiles and versions before selecting one.",
    "required": [],
    "verb": "profiles"
  },
  "projects": {
    "arguments": {},
    "fields": {},
    "order": [],
    "purpose": "List registered target projects for the GUI's target picker; the same registry as refactory projects.",
    "required": [],
    "verb": "projects"
  },
  "registry": {
    "arguments": {
      "args": {
        "default": "",
        "description": "extra registry flags, as a shell-quoted string",
        "flags": [
          "--args"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "packages": {
        "default": null,
        "description": "",
        "enum": [
          "list",
          "show",
          "verify",
          "smoke",
          "tools",
          "add",
          "import",
          "remove",
          "enable",
          "disable"
        ],
        "flags": [],
        "nargs": null,
        "position": 0,
        "positional": true,
        "required": true,
        "type": "string"
      },
      "target": {
        "default": "",
        "description": "",
        "flags": [],
        "nargs": "?",
        "position": 1,
        "positional": true,
        "required": false,
        "type": "string"
      }
    },
    "defaults": {
      "packages": "list",
      "target": ""
    },
    "fields": {
      "args": "args",
      "packages": "registry_verb",
      "target": "target"
    },
    "order": [
      "packages",
      "target",
      "args"
    ],
    "purpose": "Discover or manage mode packages using registered subcommands. Enabling/importing packages changes live discovery without rebuilding docs.",
    "required": [],
    "verb": "registry"
  },
  "run": {
    "arguments": {
      "approve": {
        "default": [],
        "description": "answer a gate a person owns, as <gate>=<value> (repeatable); a walker run stops at a user gate and prints the steer command; answer it, then resume",
        "flags": [
          "--verdict"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "intent": {
        "default": "",
        "description": "what this run is for, in your words",
        "flags": [
          "--intent"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "knobs": {
        "default": [],
        "description": "set one of the mode's knobs for this run (repeatable)",
        "flags": [
          "--knob"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "mode": {
        "default": null,
        "description": "the mode, as <package>/<name>",
        "flags": [],
        "nargs": null,
        "position": 0,
        "positional": true,
        "required": true,
        "type": "string"
      },
      "model": {
        "default": "",
        "description": "model id for this run",
        "flags": [
          "--model"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "provider": {
        "default": "",
        "description": "provider route for this run's model",
        "flags": [
          "--provider"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "approve": "verdict",
      "intent": "intent",
      "knobs": "knob",
      "mode": "mode",
      "model": "model",
      "provider": "provider"
    },
    "order": [
      "mode",
      "intent",
      "provider",
      "model",
      "approve",
      "knobs"
    ],
    "purpose": "Use a mode once. For a person present prefer wf_open so runtime user gates can ask; console run executes headlessly and reports durable output.",
    "required": [
      "mode"
    ],
    "verb": "run"
  },
  "runs": {
    "arguments": {
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "runsRoot": "runs_root"
    },
    "order": [
      "runsRoot"
    ],
    "purpose": "List durable runs under the selected runs root; historical in-flight state is not permission to resume.",
    "required": [],
    "verb": "runs"
  },
  "runtime": {
    "arguments": {},
    "fields": {},
    "order": [],
    "purpose": "Inspect local configured credential readiness and current session selection before asking setup questions; this does not test authentication.",
    "required": [],
    "verb": "runtime"
  },
  "start": {
    "arguments": {
      "knobs": {
        "default": null,
        "description": "",
        "flags": [
          "--knob"
        ],
        "items": {
          "type": "string"
        },
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "array"
      },
      "mode": {
        "default": null,
        "description": "",
        "flags": [],
        "nargs": null,
        "position": 0,
        "positional": true,
        "required": true,
        "type": "string"
      },
      "runId": {
        "default": null,
        "description": "",
        "flags": [
          "--run-id"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "knobs": "knob",
      "mode": "mode",
      "runId": "run_id",
      "runsRoot": "runs_root"
    },
    "order": [
      "mode",
      "runId",
      "runsRoot",
      "knobs"
    ],
    "purpose": "Optimize a scored mode in a separate process; returns a durable run ID for status. Project defaults, profile and explicit settings resolve before launch.",
    "required": [
      "mode"
    ],
    "settings": true,
    "verb": "start"
  },
  "status": {
    "arguments": {
      "after": {
        "default": 0,
        "description": "",
        "flags": [
          "--after"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "integer"
      },
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "after": "after",
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot",
      "after"
    ],
    "purpose": "Read durable run state and incremental events using after. A live process or receipt is not proof of successful output.",
    "required": [
      "run"
    ],
    "verb": "status"
  },
  "steer": {
    "arguments": {
      "intent": {
        "default": null,
        "description": "the request this run is for",
        "flags": [
          "--seed"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "kind": {
        "default": "info",
        "description": "a constraint must be respected; the others are context",
        "enum": [
          "info",
          "correction",
          "constraint"
        ],
        "flags": [
          "--kind"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "note": {
        "default": null,
        "description": "something said while it runs",
        "flags": [
          "--note"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "scope": {
        "default": "product",
        "description": "product (default) notes become auditable requirements; process notes record run-level directives that are visible but not auditable",
        "enum": [
          "product",
          "process"
        ],
        "flags": [
          "--scope"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "source": {
        "default": "user-relayed",
        "description": "who said this: user-relayed (default) for the person's words, ceo for the CEO's own observations; CEO notes are context, not auditable requirements",
        "enum": [
          "user-relayed",
          "ceo"
        ],
        "flags": [
          "--source"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "defaults": {
      "source": "ceo"
    },
    "fields": {
      "intent": "seed",
      "kind": "kind",
      "note": "note",
      "run": "run",
      "runsRoot": "runs_root",
      "scope": "scope",
      "source": "source"
    },
    "order": [
      "run",
      "runsRoot",
      "intent",
      "note",
      "kind",
      "scope",
      "source"
    ],
    "purpose": "Read or append to the run\u2019s intent ledger. CLI/panel also expose gate answers and boundary controls; chat only exposes notes, constraints and seed.",
    "required": [
      "run"
    ],
    "verb": "steer"
  },
  "stop": {
    "arguments": {
      "run": {
        "default": null,
        "description": "a run directory or run id",
        "flags": [
          "--run"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      },
      "runsRoot": {
        "default": null,
        "description": "the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself",
        "flags": [
          "--runs-root"
        ],
        "nargs": null,
        "positional": false,
        "required": false,
        "type": "string"
      }
    },
    "fields": {
      "run": "run",
      "runsRoot": "runs_root"
    },
    "order": [
      "run",
      "runsRoot"
    ],
    "purpose": "Request an owned run to stop and inspect its final durable state; this does not resume or delete it.",
    "required": [
      "run"
    ],
    "verb": "stop"
  },
  "strategies": {
    "arguments": {},
    "fields": {},
    "order": [],
    "purpose": "Discover registered optimizer, evaluator and outer-loop names before selecting search settings.",
    "required": [],
    "verb": "strategies"
  },
  "workers": {
    "arguments": {},
    "fields": {},
    "order": [],
    "purpose": "Inspect effective project worker routing and provenance. Configure refactory.workers.json, CLI or panel; this read-only action never changes workers or the CEO model.",
    "required": [],
    "verb": "workers"
  }
} as const
export const PARAMETERS = {
  "action": {
    "description": "Read help first; docs searches/reads only shipped documentation. adopt: Publish the exact measured winner as a new package without changing the source; overfit adoption needs the explicit CLI override. artifacts: List recorded run artifacts; optionally include contents for evidence review. best: Read the measured winning state before adoption; inspect held-out validation and required artifacts. candidate: Read one candidate and its parent diff using the run and candidate IDs. comparison: Inspect all optimization trials, including failed and unmeasured candidates, plus exact resolved settings. create: Create a reusable mode package from a brief; declare task or singleShot. For user gates prefer wf_open default_pack/create; headless approvals require explicit authorization. design: Build and check software from a brief and a check command, on its own branch in a git worktree: of the repository `target` names, or of a new project. Once its reviews pass it opens a pull request (a new project's private GitHub repository is created then), unless noPr. No reusable mode is required. Prefer wf_open default_pack/design with user gates. docs: Search or read bounded sections of the complete shipped documentation by exact index IDs; omit page/query for the index. edit: Revise a reusable mode package and validate it before QA and final intent audit. allow_regression skips original execution/comparison; revised validation remains mandatory. Prefer wf_open default_pack/edit when a person is present. help: Discover live mode/operation schemas and supported interfaces before constructing a request; topic selects one capability. leaderboard: Compare measured candidate scores and parent diffs; failed/unmeasured candidates are not winners. modes: Choose from the actual installed modes, full knob constraints, nodes/roles, ports and user gates. Unscored modes must be run, not optimized. profile-resolve: Preview a scored mode\u2019s effective optimization settings, immutable contract and worker provenance without launching a search. profiles: List immutable saved optimization profiles and versions before selecting one. projects: List registered target projects for the GUI's target picker; the same registry as refactory projects. registry: Discover or manage mode packages using registered subcommands. Enabling/importing packages changes live discovery without rebuilding docs. run: Use a mode once. For a person present prefer wf_open so runtime user gates can ask; console run executes headlessly and reports durable output. runs: List durable runs under the selected runs root; historical in-flight state is not permission to resume. runtime: Inspect local configured credential readiness and current session selection before asking setup questions; this does not test authentication. start: Optimize a scored mode in a separate process; returns a durable run ID for status. Project defaults, profile and explicit settings resolve before launch. status: Read durable run state and incremental events using after. A live process or receipt is not proof of successful output. steer: Read or append to the run\u2019s intent ledger. CLI/panel also expose gate answers and boundary controls; chat only exposes notes, constraints and seed. stop: Request an owned run to stop and inspect its final durable state; this does not resume or delete it. strategies: Discover registered optimizer, evaluator and outer-loop names before selecting search settings. workers: Inspect effective project worker routing and provenance. Configure refactory.workers.json, CLI or panel; this read-only action never changes workers or the CEO model.",
    "enum": [
      "adopt",
      "artifacts",
      "best",
      "candidate",
      "comparison",
      "create",
      "design",
      "docs",
      "edit",
      "help",
      "leaderboard",
      "modes",
      "profile-resolve",
      "profiles",
      "projects",
      "registry",
      "run",
      "runs",
      "runtime",
      "start",
      "status",
      "steer",
      "stop",
      "strategies",
      "workers"
    ],
    "required": true,
    "type": "string"
  },
  "after": {
    "description": "status: Read durable run state and incremental events using after. A live process or receipt is not proof of successful output. (console after). Read help topic=status for defaults and constraints.",
    "type": "integer"
  },
  "allowRegression": {
    "description": "edit: skip original-mode execution and comparison; still validate the revision (console allow_regression). Read help topic=edit for defaults and constraints.",
    "type": "boolean"
  },
  "approve": {
    "description": "create: answer a gate a person owns, as <gate>=<value> (repeatable); gate_strategy is one, so a headless create is told the answer (console verdict). Read help topic=create for defaults and constraints. design: answer a gate a person owns, as <gate>=<value> (repeatable) (console verdict). Read help topic=design for defaults and constraints. edit: answer a gate a person owns, as <gate>=<value> (repeatable) (console verdict). Read help topic=edit for defaults and constraints. run: answer a gate a person owns, as <gate>=<value> (repeatable); a walker run stops at a user gate and prints the steer command; answer it, then resume (console verdict). Read help topic=run for defaults and constraints.",
    "items": {
      "type": "string"
    },
    "type": "array"
  },
  "args": {
    "description": "registry: extra registry flags, as a shell-quoted string (console args). Read help topic=registry for defaults and constraints.",
    "type": "string"
  },
  "as": {
    "description": "adopt: the new package name (console as_name). Read help topic=adopt for defaults and constraints.",
    "type": "string"
  },
  "attempts": {
    "description": "create: Create a reusable mode package from a brief; declare task or singleShot. For user gates prefer wf_open default_pack/create; headless approvals require explicit authorization. (console attempts). Read help topic=create for defaults and constraints. design: Build and check software from a brief and a check command, on its own branch in a git worktree: of the repository `target` names, or of a new project. Once its reviews pass it opens a pull request (a new project's private GitHub repository is created then), unless noPr. No reusable mode is required. Prefer wf_open default_pack/design with user gates. (console attempts). Read help topic=design for defaults and constraints. edit: Revise a reusable mode package and validate it before QA and final intent audit. allow_regression skips original execution/comparison; revised validation remains mandatory. Prefer wf_open default_pack/edit when a person is present. (console attempts). Read help topic=edit for defaults and constraints.",
    "type": "integer"
  },
  "base": {
    "description": "design: the commit, branch or tag the run starts from (console base). Read help topic=design for defaults and constraints.",
    "type": "string"
  },
  "brief": {
    "description": "create: Create a reusable mode package from a brief; declare task or singleShot. For user gates prefer wf_open default_pack/create; headless approvals require explicit authorization. (console brief). Read help topic=create for defaults and constraints. design: Build and check software from a brief and a check command, on its own branch in a git worktree: of the repository `target` names, or of a new project. Once its reviews pass it opens a pull request (a new project's private GitHub repository is created then), unless noPr. No reusable mode is required. Prefer wf_open default_pack/design with user gates. (console brief). Read help topic=design for defaults and constraints. edit: Revise a reusable mode package and validate it before QA and final intent audit. allow_regression skips original execution/comparison; revised validation remains mandatory. Prefer wf_open default_pack/edit when a person is present. (console brief). Read help topic=edit for defaults and constraints.",
    "type": "string"
  },
  "candidate": {
    "description": "candidate: Read one candidate and its parent diff using the run and candidate IDs. (console candidate). Read help topic=candidate for defaults and constraints.",
    "type": "string"
  },
  "check": {
    "description": "design: the command that proves the artifact works (console check). Read help topic=design for defaults and constraints.",
    "type": "string"
  },
  "contents": {
    "description": "artifacts: List recorded run artifacts; optionally include contents for evidence review. (console contents). Read help topic=artifacts for defaults and constraints.",
    "type": "boolean"
  },
  "dataNode": {
    "description": "create: build the mode as a DataNode that iterates its subgraph per item (console data_node). Read help topic=create for defaults and constraints.",
    "type": "boolean"
  },
  "extends": {
    "description": "create: Create a reusable mode package from a brief; declare task or singleShot. For user gates prefer wf_open default_pack/create; headless approvals require explicit authorization. (console extends). Read help topic=create for defaults and constraints.",
    "type": "string"
  },
  "generate": {
    "description": "design: a command run in the worktree after dependencies are installed, before setup and each check; whatever it changes in tracked files is amended into the attempt. A generator that itself needs a build includes that build in its own command (console generate). Read help topic=design for defaults and constraints.",
    "type": "string"
  },
  "headless": {
    "description": "Required true with approve: caller explicitly says nobody is at the keyboard; otherwise use wf_open so the runtime asks user gates.",
    "type": "boolean"
  },
  "inputs": {
    "description": "create: an input the produced mode must accept, named the way a knob or a file would be (repeatable); validation refuses a mode with no way to receive it (console input). Read help topic=create for defaults and constraints.",
    "items": {
      "type": "string"
    },
    "type": "array"
  },
  "intent": {
    "description": "run: what this run is for, in your words (console intent). Read help topic=run for defaults and constraints. steer: the request this run is for (console seed). Read help topic=steer for defaults and constraints.",
    "type": "string"
  },
  "kind": {
    "description": "steer: a constraint must be respected; the others are context (console kind). Read help topic=steer for defaults and constraints.",
    "enum": [
      "info",
      "correction",
      "constraint"
    ],
    "type": "string"
  },
  "knobs": {
    "description": "profile-resolve: Preview a scored mode\u2019s effective optimization settings, immutable contract and worker provenance without launching a search. (console knob). Read help topic=profile-resolve for defaults and constraints. run: set one of the mode's knobs for this run (repeatable) (console knob). Read help topic=run for defaults and constraints. start: Optimize a scored mode in a separate process; returns a durable run ID for status. Project defaults, profile and explicit settings resolve before launch. (console knob). Read help topic=start for defaults and constraints.",
    "items": {
      "type": "string"
    },
    "type": "array"
  },
  "limit": {
    "description": "docs: maximum text characters, 1..12000 (console limit). Read help topic=docs for defaults and constraints. leaderboard: Compare measured candidate scores and parent diffs; failed/unmeasured candidates are not winners. (console limit). Read help topic=leaderboard for defaults and constraints.",
    "type": "integer"
  },
  "mode": {
    "description": "edit: Revise a reusable mode package and validate it before QA and final intent audit. allow_regression skips original execution/comparison; revised validation remains mandatory. Prefer wf_open default_pack/edit when a person is present. (console mode). Read help topic=edit for defaults and constraints. profile-resolve: Preview a scored mode\u2019s effective optimization settings, immutable contract and worker provenance without launching a search. (console mode). Read help topic=profile-resolve for defaults and constraints. run: the mode, as <package>/<name> (console mode). Read help topic=run for defaults and constraints. start: Optimize a scored mode in a separate process; returns a durable run ID for status. Project defaults, profile and explicit settings resolve before launch. (console mode). Read help topic=start for defaults and constraints.",
    "type": "string"
  },
  "model": {
    "description": "create: model id for this authoring run (console model). Read help topic=create for defaults and constraints. design: model id for this authoring run (console model). Read help topic=design for defaults and constraints. edit: model id for this authoring run (console model). Read help topic=edit for defaults and constraints. run: model id for this run (console model). Read help topic=run for defaults and constraints.",
    "type": "string"
  },
  "name": {
    "description": "create: Create a reusable mode package from a brief; declare task or singleShot. For user gates prefer wf_open default_pack/create; headless approvals require explicit authorization. (console name). Read help topic=create for defaults and constraints. design: Build and check software from a brief and a check command, on its own branch in a git worktree: of the repository `target` names, or of a new project. Once its reviews pass it opens a pull request (a new project's private GitHub repository is created then), unless noPr. No reusable mode is required. Prefer wf_open default_pack/design with user gates. (console name). Read help topic=design for defaults and constraints. edit: Revise a reusable mode package and validate it before QA and final intent audit. allow_regression skips original execution/comparison; revised validation remains mandatory. Prefer wf_open default_pack/edit when a person is present. (console name). Read help topic=edit for defaults and constraints.",
    "type": "string"
  },
  "noPr": {
    "description": "design: keep the work on the run's local branch instead of opening a pull request (console no_pr). Read help topic=design for defaults and constraints.",
    "type": "boolean"
  },
  "noWorktree": {
    "description": "design: write the artifact under .factory/artifacts instead of a worktree (console no_worktree). Read help topic=design for defaults and constraints.",
    "type": "boolean"
  },
  "note": {
    "description": "steer: something said while it runs (console note). Read help topic=steer for defaults and constraints.",
    "type": "string"
  },
  "offset": {
    "description": "docs: character offset within page or section (console offset). Read help topic=docs for defaults and constraints.",
    "type": "integer"
  },
  "packages": {
    "description": "registry: Discover or manage mode packages using registered subcommands. Enabling/importing packages changes live discovery without rebuilding docs. (console registry_verb). Read help topic=registry for defaults and constraints.",
    "enum": [
      "list",
      "show",
      "verify",
      "smoke",
      "tools",
      "add",
      "import",
      "remove",
      "enable",
      "disable"
    ],
    "type": "string"
  },
  "page": {
    "description": "docs: exact page ID from the docs index (console page). Read help topic=docs for defaults and constraints.",
    "type": "string"
  },
  "provider": {
    "description": "create: provider route for this authoring run (console provider). Read help topic=create for defaults and constraints. design: provider route for this authoring run (console provider). Read help topic=design for defaults and constraints. edit: provider route for this authoring run (console provider). Read help topic=edit for defaults and constraints. run: provider route for this run's model (console provider). Read help topic=run for defaults and constraints.",
    "type": "string"
  },
  "query": {
    "description": "docs: words to search in shipped documentation (console query). Read help topic=docs for defaults and constraints.",
    "type": "string"
  },
  "refine": {
    "description": "design: an open pull request to follow up (its URL or owner/repo#number); the run starts from its branch and updates it (console refine). Read help topic=design for defaults and constraints.",
    "type": "string"
  },
  "resume": {
    "description": "create: continue a failed run from its directory (console resume). Read help topic=create for defaults and constraints. design: continue a failed run from its directory (console resume). Read help topic=design for defaults and constraints. edit: continue a failed run from its directory (console resume). Read help topic=edit for defaults and constraints.",
    "type": "string"
  },
  "reviewComments": {
    "description": "design: with --refine: add the pull request's reviews and comments to the steering (console review_comments). Read help topic=design for defaults and constraints.",
    "type": "boolean"
  },
  "run": {
    "description": "adopt: a run directory or run id (console run). Read help topic=adopt for defaults and constraints. artifacts: a run directory or run id (console run). Read help topic=artifacts for defaults and constraints. best: a run directory or run id (console run). Read help topic=best for defaults and constraints. candidate: a run directory or run id (console run). Read help topic=candidate for defaults and constraints. comparison: a run directory or run id (console run). Read help topic=comparison for defaults and constraints. leaderboard: a run directory or run id (console run). Read help topic=leaderboard for defaults and constraints. status: a run directory or run id (console run). Read help topic=status for defaults and constraints. steer: a run directory or run id (console run). Read help topic=steer for defaults and constraints. stop: a run directory or run id (console run). Read help topic=stop for defaults and constraints.",
    "type": "string"
  },
  "runId": {
    "description": "start: Optimize a scored mode in a separate process; returns a durable run ID for status. Project defaults, profile and explicit settings resolve before launch. (console run_id). Read help topic=start for defaults and constraints.",
    "type": "string"
  },
  "runsRoot": {
    "description": "adopt: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=adopt for defaults and constraints. artifacts: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=artifacts for defaults and constraints. best: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=best for defaults and constraints. candidate: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=candidate for defaults and constraints. comparison: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=comparison for defaults and constraints. leaderboard: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=leaderboard for defaults and constraints. runs: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=runs for defaults and constraints. start: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=start for defaults and constraints. status: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=status for defaults and constraints. steer: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=steer for defaults and constraints. stop: the directory runs live under, one subdirectory per run; this is not the loop's --runs-root, which is the run directory itself (console runs_root). Read help topic=stop for defaults and constraints.",
    "type": "string"
  },
  "sample": {
    "description": "create: a knob value the validation run should use (repeatable), so the smoke exercises the input path rather than the built-in fixture (console sample). Read help topic=create for defaults and constraints. edit: a knob value the validation run should use (repeatable) (console sample). Read help topic=edit for defaults and constraints.",
    "items": {
      "type": "string"
    },
    "type": "array"
  },
  "scope": {
    "description": "steer: product (default) notes become auditable requirements; process notes record run-level directives that are visible but not auditable (console scope). Read help topic=steer for defaults and constraints.",
    "enum": [
      "product",
      "process"
    ],
    "type": "string"
  },
  "section": {
    "description": "docs: section ID from the docs index (console section). Read help topic=docs for defaults and constraints.",
    "type": "string"
  },
  "settings": {
    "additionalProperties": true,
    "description": "Explicit start/profile-resolve settings from help topic=start. Uses actual console flags; JSON route maps are supported. Profile and project defaults apply to omitted fields.",
    "type": "object"
  },
  "setup": {
    "description": "design: a command run in the worktree before each check (console setup). Read help topic=design for defaults and constraints.",
    "type": "string"
  },
  "singleShot": {
    "description": "create: build a harness for one run of one thing, with no Task binding; one of --task or --single is required (console single_shot). Read help topic=create for defaults and constraints.",
    "type": "boolean"
  },
  "source": {
    "description": "steer: who said this: user-relayed (default) for the person's words, ceo for the CEO's own observations; CEO notes are context, not auditable requirements (console source). Read help topic=steer for defaults and constraints.",
    "enum": [
      "user-relayed",
      "ceo"
    ],
    "type": "string"
  },
  "state": {
    "description": "best: include the full state (console state). Read help topic=best for defaults and constraints.",
    "type": "boolean"
  },
  "target": {
    "description": "design: the git repository to change, in a worktree of its own (default: a new project) (console target). Read help topic=design for defaults and constraints. registry: Discover or manage mode packages using registered subcommands. Enabling/importing packages changes live discovery without rebuilding docs. (console target). Read help topic=registry for defaults and constraints.",
    "type": "string"
  },
  "task": {
    "description": "create: the class of work: a Task the produced harness binds a data node to, so the harness runs its subgraph once per item the Task lists (console task_ref). Read help topic=create for defaults and constraints.",
    "type": "string"
  },
  "topic": {
    "description": "help: operation or mode ID; omit for the live index (console topic). Read help topic=help for defaults and constraints.",
    "type": "string"
  }
} as const
export const OPERATING_GUIDE = "# CEO operating guide\n\nRefactory runs explicit graphs. The runtime owns node order, branches, checks,\nevidence, user gates and recovery. Drive the selected graph; do not reproduce its\nworkers in conversation or mark work complete from prose alone.\n\nStart with `refactory` action `help`. Read `help` with `topic` equal to a mode ID\nor operation for its live arguments, defaults, constraints, examples and interface\nroutes. `modes` lists installed packages, including packages added since build.\nCoverage warnings mean guidance may be missing or stale. Read `docs` with `query`\nfor search, or `page`, optional `section`, `offset` and `limit` for bounded reads.\nUse the returned IDs and continuation offsets. Help and docs carry schema, build\nand documentation identities. These APIs read shipped product docs; no shell,\nsource-code search or web access is needed for product discovery.\n\nCreate and Edit produce reusable mode packages. Design produces a software\nartifact. Canonical validation precedes QA and final intent review. Fast variants retain\nfocused QA and bounded repair. Edit allow_regression skips original execution and\ncomparison, while revised validation remains mandatory. Run uses a mode\nonce; Optimize (`start`) searches a scored mode. Read the mode's ports, required\ninputs and user gates before choosing. When a person is present, use `wf_open`\nwith the mode, knobs, intent and a fresh absolute workdir. Continue with the\nworkflow tools. The runtime asks user gates; never invent their verdicts.\nHeadless approvals require an explicit user-approved answer and `headless: true`.\n\nFor optimization, inspect `strategies` and `profiles`; `profile-resolve` previews\nsettings and the contract. `start` returns a durable run ID. Read incremental\n`status` using its offset, then candidates, artifacts and winner evidence. A run\ncan outlive a chat. A stopped, failed or unmeasured run is not success. Recovery\nrequires a known saved boundary; use the documented CLI or panel recovery route.\n\n`workers` explains effective worker routing and provenance. New runs load\n`refactory.workers.json` at the project root. Explicit profile/launch settings\nwin; node overrides beat role overrides, which beat the run route. This file\nselects graph workers, not this CEO's conversational model. Backend changes are\nconfigured through the file, CLI or panel. Native authentication stays in its\nexisting stores. New worker permissions default to `skip`; token caps are unset\nunless requested, and native workers do not support whole-tree token caps.\n\nUse `runtime` for configured credential readiness, which is not authentication\nproof. Follow explicit task and fixture constraints without asking again. Keep\nbuild/docs identity with evidence, and distinguish a generated receipt from\nchecks that establish the requested output.\n\n## Delegation and review discipline\n\nYou are an orchestrator. Delegate all technical work \u2014 writing code, fixing\nbugs, running checks, performing research \u2014 to the graph's agent nodes via\nsubagents. Review their output. When an agent fails, retry with better\ninstructions (wf_edit set-prompt/append-prompt, then wf_redo) or stop. Never\ntake over the agent's work yourself. For nodes whose prompts contain template\nplaceholders (double-brace variables), prefer `append-prompt`; `set-prompt` is\nrefused when the replacement drops a placeholder.\n\nEvery change that produces a pull request must pass through the graph's review\nnodes. \"The change\nis small\" or \"the code is already correct\" is not a reason to skip them.\n\nNever claim, restate, or re-apply a person's verdict or approval in a ledger\nnote or any other message. A user gate is answered only by the person through\nthe runtime; a ledger note about a gate verdict is your observation, not an\nanswer. Never use wf_edit skip on a check node (a node that produces runtime\nevidence). When a check cannot run (sandbox limit, missing tool, infrastructure\nstop), report what stopped it and what would unblock it\n(authoring_evidence.stop_unrepairable names the cause). When the person wants\nto skip the check, give them the exact command:\n`refactory graph-control skip --run <dir> --node <check_node>`.\nWhen the owner has stopped, the person can resume with\n`refactory graph-control resume --run <dir> [--acknowledge-failure]`.\nIf the person skips, the run report and PR say the check did not run.\n" as const
export const BUILD_ID = "4016f0c0c743465995c96ee8b190d78614403e4c3b4c38575e648c627d32c64d" as const
