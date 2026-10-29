#!/usr/bin/env bash
# Validate patch syntax even when a warm pnpm store reuses package artifacts.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  set -- "$root"/patches/*.patch
fi
for patch in "$@"; do
  git apply --numstat "$patch" > /dev/null
done
