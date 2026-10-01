#!/usr/bin/env bash
# refactory-dsh — one-command install for this repository.
#
# The repository owns eight TypeScript packages and depends on the published
# `@deepseek-ai/dsh` pin for the harness; the Python glue lives at `refactory/`.
# This script sets up the Node side (pnpm) and the Python side (one `uv sync`
# over the root project), so after it:
#
#   uv run refactory doctor      is this machine ready?
#   uv run refactory list        what can this checkout run?
#   uv run refactory ceo "an idea" --mode design
#
#   git clone https://github.com/akashgit/refactory-dsh.git
#   cd refactory-dsh
#   ./install.sh
#   uv run refactory ceo "a mode that drafts a haiku and counts its syllables" --mode create
#
# It provisions the default install and nothing optional:
#
#   * the pinned pnpm workspace and the eight packages' built `lib/`;
#   * `.venv` from the root `pyproject.toml` — the default (`dev`) dependency
#     group only;
#   * the two DSH profiles this checkout owns — `graph`, which the Python glue
#     runs modes on, and `refactory-web`, which the GUI boots over the harness's
#     own `web` template (the harness's `web` is never written to) — idempotently
#     (`REFACTORY_SKIP_DSH_PROFILE=1` skips both).
#
# It deliberately leaves out the two opt-in, non-shipping pieces, and the CI gate in
# `.github/workflows/ci.yml` asserts this default install rather than them:
#
#   * the non-default `reference` group (remote-factory, providing `factory`/`srf`),
#     a test-time oracle added with `uv sync --group reference`. Without it the
#     parity tests skip with a reason rather than fail.
#   * `refactory/vendor/srf/graphs`, the legacy graph directory for the low-level
#     `dsh --profile graph run` path. The public `uv run refactory ceo
#     <package>/<mode>` route lowers the mode and supplies its own graph directory,
#     so a fresh checkout needs neither.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

# Node 22 is required (engines: ^22.19 || >=24); prefer the nvm install when
# present so `node` resolves to a supported version regardless of PATH.
if [ -x "$HOME/.nvm/versions/node/v22.23.2/bin/node" ]; then
  export PATH="$HOME/.nvm/versions/node/v22.23.2/bin:$PATH"
fi

echo "== refactory-dsh install =="
echo "  repo:  $HERE"

# ── prerequisites ────────────────────────────────────────────────────────────
# The script satisfies what it can instead of asking the reader to: Node has to be
# there (it is a runtime, not a shim), but pnpm is pinned by `packageManager` in
# package.json, and corepack can run that pin *without* the reader enabling anything.
#
# The failure this exists for: a machine with Node 22 and corepack but no `pnpm` on
# PATH stopped at "pnpm not found — install Node >=22.19 + corepack", which is advice
# the reader had already followed.
if ! command -v node >/dev/null 2>&1; then
  echo "  error: no node on PATH. Install Node >=22.19 (https://nodejs.org, or nvm:" >&2
  echo "         nvm install 22 && nvm use 22) and run this script again." >&2
  exit 1
fi
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
NODE_MINOR="$(node -p 'process.versions.node.split(".")[1]')"
if [ "$NODE_MAJOR" -lt 22 ] || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -lt 19 ]; }; then
  echo "  error: node $(node --version) is too old; this workspace needs >=22.19 or >=24." >&2
  echo "         nvm install 22 && nvm use 22" >&2
  exit 1
fi
echo "  node:  $(node --version)"
# The even releases are the ones this workspace is tested on. An odd major is a
# current release, not LTS, and it may be younger than the packages here — worth
# saying before a build failure rather than after one.
if [ $((NODE_MAJOR % 2)) -eq 1 ] && [ "$NODE_MAJOR" -gt 22 ]; then
  echo "  note:  node $(node --version) is a current release; 22 and 24 are what this workspace is tested on." >&2
  echo "         If the build below fails, try: nvm install 22 && nvm use 22" >&2
fi

# The pnpm version this workspace pins, read from its own manifest.
PNPM_PIN="$(sed -n 's/.*"packageManager"[[:space:]]*:[[:space:]]*"pnpm@\([^"]*\)".*/\1/p' "$HERE/package.json" | head -1)"
PNPM_PIN="${PNPM_PIN:-latest}"

# Get `pnpm` onto PATH, by whichever route this machine allows, in the order that
# asks the least of the reader:
#
#   1. the `pnpm` already on PATH;
#   2. corepack, which runs package.json's pin with nothing enabled — but corepack is
#      *not* bundled from Node 25 on, so it is not the reliable one;
#   3. `npm install -g pnpm@<pin>`, which needs only the npm that ships with Node.
#
# It must end up as a real `pnpm` on PATH and not merely as an array this script
# expands: the build runs `pnpm run build:lib:host` from *inside* another pnpm
# script, so a bare `pnpm` has to resolve in a child shell too. (That is what broke
# the first version of this: `sh: pnpm: command not found` from the build's own
# subshell.) `npx pnpm` is deliberately absent — when it has to download the package
# rather than use a warm cache, its bin re-executes `pnpm` by name and fails.
PNPM_DIR=""
if ! command -v pnpm >/dev/null 2>&1; then
  if command -v corepack >/dev/null 2>&1; then
    echo "  pnpm:  not on PATH; using corepack with the pinned pnpm@$PNPM_PIN"
    PNPM_DIR="$(mktemp -d)"
    printf '#!/bin/sh\nexec corepack pnpm "$@"\n' > "$PNPM_DIR/pnpm"
    chmod +x "$PNPM_DIR/pnpm"
  elif [ "${REFACTORY_SKIP_GLOBAL_PNPM:-}" != "1" ] && command -v npm >/dev/null 2>&1; then
    echo "  pnpm:  not on PATH and no corepack; installing pnpm@$PNPM_PIN with npm"
    echo "         (skip this with REFACTORY_SKIP_GLOBAL_PNPM=1)"
    npm install -g "pnpm@$PNPM_PIN"
    # The global prefix is not always on PATH — an nvm or homebrew Node puts its bin
    # there, a system Node does not — so use the directory npm itself reports.
    PNPM_DIR="$(npm prefix -g)/bin"
  else
    echo "  error: no way to run pnpm: neither pnpm, corepack, nor npm is on PATH." >&2
    echo "         Install it once, then run this script again:" >&2
    echo "           npm install -g pnpm@$PNPM_PIN" >&2
    exit 1
  fi
fi
if [ -n "$PNPM_DIR" ]; then
  export PATH="$PNPM_DIR:$PATH"
fi
command -v pnpm >/dev/null 2>&1 || {
  echo "  error: pnpm is still not runnable after installing it. Try:" >&2
  echo "           npm install -g pnpm@$PNPM_PIN" >&2
  exit 1
}
echo "  pnpm:  $(pnpm --version)"

# 1. Python side first: **one project, one lockfile, one command**. `uv sync`
#    reads the root `pyproject.toml`, creates `.venv`, installs the local
#    `refactory-dsh` package in editable mode and puts the `refactory` console
#    script on `.venv/bin` — which is why every documented command is
#    `uv run refactory …` rather than an interpreter path.
#
#    This must precede the Node build: `pnpm run build` starts with `docs:check`,
#    which runs `.venv/bin/python`, so on a checkout with no `.venv` the build
#    failed before `uv sync` ever ran and `set -euo pipefail` stopped the install.
#    Ordering is the fix; the build genuinely needs the interpreter the docs
#    generator runs on.
if ! command -v uv >/dev/null 2>&1; then
  echo "  error: uv not found. Install it with:" >&2
  echo "           curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "         (or: brew install uv, or pipx install uv) and run this script again." >&2
  exit 1
fi
(cd "$HERE" && uv sync)

# 2. Node side: install the pnpm workspace (the pinned harness plus the eight
#    packages this repo owns), then build those packages' lib/ output. It depends
#    on step 1 only through `docs:check`, which is why the Python side is first.
(cd "$HERE" && pnpm install --frozen-lockfile && pnpm run build)

# 3. the DSH profiles this checkout runs on. A profile is **machine state** — it
#    is created by `dsh plugin --profile <name> add <package>`, not derived from
#    the bundle — so a fresh clone that skips this can run every mode through the
#    walker and still fail the first time anything reaches DSH. The launcher is
#    the pinned harness the root manifest installed, not a copy in this repo.
#    This checkout owns two profiles: `graph`, which the Python glue runs modes
#    on, and `refactory-web`, which the GUI boots over the harness's own `web`
#    template. The harness's `web` profile is never written to. Without the
#    preset layer every session the GUI creates without naming one stops with
#    `agent-preset/not-found`, and without the surface layer the console and
#    picker rows never mount. The plan comes from `dsh_profile`, so this script
#    cannot drift from the check `doctor` makes. Idempotent: a profile that
#    already carries a layer is left alone. `REFACTORY_SKIP_DSH_PROFILE=1` skips
#    it for someone who only wants the Python side and does not want ~/.dsh touched.
DSH="$HERE/node_modules/.bin/dsh"
if [ "${REFACTORY_SKIP_DSH_PROFILE:-}" = "1" ]; then
  echo "  DSH profiles: skipped (REFACTORY_SKIP_DSH_PROFILE=1)"
else
  # The plan is `dsh_profile.provision_plan()`: `init` copies a shipped template's
  # bundle list, `add` installs one bundle from the checkout. Executing its rows
  # rather than iterating a second copy of the profile list in shell is what keeps
  # the composition in one place.
  PLAN="$("$VENV/bin/python" -c 'from refactory_dsh import dsh_profile; print(dsh_profile.render_plan(), end="")')"
  if [ -z "$PLAN" ]; then
    echo "  DSH profiles: already present"
  else
    while IFS=$'\t' read -r kind profile value; do
      [ -n "$kind" ] || continue
      case "$kind" in
        init)
          echo "  DSH profile '$profile': creating from the '$value' template (machine state, once per machine)"
          "$DSH" --profile "$profile" --from-default-profile "$value" --dump-config >/dev/null
          ;;
        add)
          echo "  DSH profile '$profile': adding the '$value' layer"
          "$DSH" plugin --profile "$profile" add "$HERE/packages/bundle/$value"
          ;;
      esac
    done <<<"$PLAN"
  fi
fi

# 4. say what happened, in the words the docs use.
echo
"$VENV/bin/refactory" doctor || true

echo
echo "  done. next:"
echo "    uv run refactory list"
echo "    uv run refactory ceo \"an idea\" --mode design"
echo "    uv run refactory ui          # or: $HERE/refactory-dsh --port 3081 --no-open"
