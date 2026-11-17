#!/usr/bin/env bash
# Run the sealed-step tests on this machine, the hostile step among them, and fail
# (never skip) where the seal cannot run. Usage, from a checkout with .venv synced:
#   scripts/sealed-hostile-check.sh [checkout]
# Run it from a login shell, so GOOGLE_APPLICATION_CREDENTIALS is the real one the
# hostile step tries to read; on a machine whose model proxy listens on
# 127.0.0.1:8082, the step tries that proxy itself.
set -uo pipefail
root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$root" || exit 2
echo "checkout: $root ($(git rev-parse --short HEAD 2>/dev/null))"
echo "kernel: $(uname -r); max_user_namespaces: $(cat /proc/sys/user/max_user_namespaces 2>&1)"
if out=$(unshare -Urnm true 2>&1); then echo "unshare -Urnm: works"; else echo "unshare -Urnm: FAILS: $out"; fi
REFACTORY_REQUIRE_CODE_SANDBOX=1 .venv/bin/python -m pytest -q -rs -p no:cacheprovider \
  refactory/tests/test_sealed_tools.py
