#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

make test-family-repair-contracts test-first-divergence test-float-compare

if [ "$(uname -s)" = Darwin ]; then
    make test-projection-repairs-metal
else
    echo "PROJECTION_REPAIR_METAL result=SKIP reason=requires_Darwin_Metal"
fi
