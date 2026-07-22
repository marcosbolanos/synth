#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

git -C "${repo_root}" submodule update --init --recursive src/vital

uv run -- make -C "${repo_root}/src/vital" \
  headless_server \
  CONFIG=Release \
  -j"$(nproc)"

echo "Vital headless renderer built at:"
echo "${repo_root}/src/vital/headless/builds/linux/build/vital"
