---
description: "The refactory agent-preset bundle: ships the `ceo` preset from this package's own tree through the agent-presets roster's configured roots."
kind: "package-bundle"
---

# @deepseek-ai/dsh-refactory-presets

## Summary

This bundle is one patch layer plus a startup plugin. The patch points the agent-presets roster at the `presets/` directory beside this package's own `lib/`, using the roster's configured-roots mechanism — a deployment-supplied `system` root — and sets `ceo` as the default preset. Nothing here edits `@deepseek-ai/dsh-agent-presets`: the `ceo` composition and its `preset.yml` display metadata live in this package and are read from it. The layer is **surface-independent**: install it into any profile whose sessions should resolve `CEO` from this checkout, browser or TUI. The browser console is a separate layer, `@deepseek-ai/dsh-refactory-web`.

## Use this package

### Install into a profile

```text
# the re:factory-owned GUI profile (harness web surface + this layer)
dsh plugin --profile refactory-web add <checkout>/packages/bundle/refactory-presets
# or attach the surface-independent layer to a profile you already have (e.g. Martty)
dsh plugin --profile martty add <checkout>/packages/bundle/refactory-presets
dsh plugin --profile refactory-web remove @deepseek-ai/dsh-refactory-presets
```

`@deepseek-ai/dsh-refactory-presets` is not published to the npm registry, so `add` takes a path into a checkout, and the profile links it. The manifest declares `dsh.bundle.patch`, so the reconcile step appends `@deepseek-ai/dsh-refactory-presets` to `dsh.profile.bundles` and the layer activates on the next boot.

### Layer-ordering precondition

The `agent-presets` row this patch overrides is inserted by the `dsh-web-app` layer, and a patch only reaches a row that an earlier layer created. **This bundle must therefore be applied after `@deepseek-ai/dsh-web-app` in `dsh.profile.bundles`.** `dsh plugin add` appends bundles in dependency order, so an ordinary profile install satisfies that automatically; only a hand-edited bundle list needs to keep the order.

### What you get

- The `ceo` preset, with the display name `CEO` and roster `order: 1`, discovered as a `system`-trust row from this package's `presets/` tree.
- `ceo` as the roster default, so a session created without a named preset composes from it.
- The interactive workflow spine that the preset mounts (`wf_*` tools, `interactiveOpen`), with its `ceoDeny` list leaving the CEO on the workflow tools rather than the coding tools.

### Verify it, without trusting this README

```bash
pnpm run test:acceptance     # packages/bundle/refactory-presets/tests/verify-ceo-root.mjs
```

`tests/startup.spec.ts` is the fast unit check; the acceptance run is the one that proves the mechanism, because only it can see the layer order and the roster the harness actually builds. It runs `dsh_profile`'s provisioning plan into a throwaway `DSH_HOME`, composes the owned `refactory-web` profile from the harness's `web` template, checks the harness's own `web` template keeps exactly its template bundles and mounts none of re:factory's rows, then boots `refactory-web` and asks its live services: the roster row (`ceo`, `trust: system`, name `CEO`, not broken), `session/create` without a preset, and the composed agent's own tool catalog (`wf_*` present, the preset's `ceoDeny` absent). It also reads the **installed** `@deepseek-ai/dsh-agent-presets` tree and fails if that tree still ships a `ceo` directory — without that check a stale patched `node_modules` would satisfy every live assertion with the configured root doing nothing. CI runs it as the `Web profile acceptance` step.

## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

A patch row is evaluated as `new Function('ctx', 'expr', ...)`, so a `!!js` expression has no `import.meta` and cannot compute its own package directory. This bundle therefore splits the work: the `refactory-presets-startup` row mounts a small plugin that resolves `presets/` beside its own module and publishes it as the `refactoryPresetsStartup` service; the `agent-presets` row injects that service and reads `ctx.refactoryPresetsStartup.presetRoot` as its configured root path.

`agent-presets` replaces a row's whole `config` object rather than merging it, so the patch restates `default` (the schema requires it) and supplies `roots`; `includeShippedRoot` and `includeUserRoot` keep their `true` defaults, so the shipped and user presets stay mounted exactly as before.

That replacement rule makes this row's `config` **single-owner**. A layer applied *after* this bundle that patches the same `id` without repeating `roots` silently drops this bundle's root — the `ceo` preset then vanishes, and the only symptom is an `agent-preset/not-found` at session create. Which layer is last follows `dsh.profile.bundles` order, so a second patcher of `agent-presets` is a conflict rather than a merge. The `dsh-web-app` layer owns the row's other keys and, now that this bundle supplies `roots`, must not state `agent-presets` at all.

### Source map

| File | Role |
|---|---|
| [`cordis.patch.yml`](cordis.patch.yml) | The layer: the startup row, then the `agent-presets` override |
| [`src/startup.ts`](src/startup.ts) | The `refactory-presets-startup` provider: resolves and publishes `presets/` |
| [`presets/ceo/agent.cordis.yml`](presets/ceo/agent.cordis.yml) | The CEO composition, mounted once per process by the roster |
| [`presets/ceo/preset.yml`](presets/ceo/preset.yml) | Display metadata: name `CEO`, description, roster `order` |

</details>

## Known Limitations and Deferred Work

- **The bundle carries the `ceo` composition only.** Any other refactory-owned preset added later belongs in the same `presets/` tree; the startup provider publishes the directory, not a preset.
- **The composition's plugin packages must resolve from the profile.** The roster resolves a preset row's package names against the base the harness was loaded from, not against this package, so the profile's install must carry the packages `ceo` names.
- **Layer order is a precondition, not an assertion.** A patch only reaches a row an earlier layer created, so a profile that lists this bundle *before* `@deepseek-ai/dsh-web-app` gets a `patch: entry "agent-presets" not found` warning and no default is set at all — not a fallback to a shipped preset. On a profile that carries this layer but no `agent-presets` row, the composition still names `ceo` as the roster default, so a session created without naming one then stops with `agent-preset/not-found`; on a profile with no `agent-presets` row at all there is simply no default. `dsh plugin add` appends bundles in dependency order, so an ordinary profile install gets the order right, and `dsh_profile`'s target table is what `./install.sh`, `./refactory-dsh` and `refactory doctor` use to notice a profile that lacks the layer entirely.

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

`ceo` was previously delivered by patching `@deepseek-ai/dsh-agent-presets` to carry `presets/ceo/agent.cordis.yml` and `presets/ceo/preset.yml` in its published tree. That patch is gone; the two files are the same bytes, moved into this package. If you change a preset, change it here.

</details>
