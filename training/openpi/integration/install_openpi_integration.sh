#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
openpi_root="${1:?usage: $0 /path/to/openpi}"
expected_commit="215abfb217dbac7d5f1273282331b9b1866c0479"

actual_commit="$(git -C "$openpi_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "OpenPI commit mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 1
fi

install -m 0644 "$script_dir/fr3_policy.py" "$openpi_root/src/openpi/policies/fr3_policy.py"
if git -C "$openpi_root" apply --reverse --check "$script_dir/config.patch" 2>/dev/null; then
    echo "OpenPI FR3 config patch is already applied."
elif git -C "$openpi_root" apply --check "$script_dir/config.patch"; then
    git -C "$openpi_root" apply "$script_dir/config.patch"
    echo "Applied OpenPI FR3 config patch."
else
    echo "Cannot apply FR3 config patch cleanly" >&2
    exit 1
fi
