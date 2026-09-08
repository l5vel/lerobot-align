#!/usr/bin/env bash
# Exercise installed distributions without importing the development checkout.
set -euo pipefail
release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
release_check=$(mktemp -d "${TMPDIR:-/tmp}/lerobot-align-release.XXXXXX")
export HF_HOME="$release_check/hf"
export HF_DATASETS_CACHE="$release_check/hf/datasets"
export HF_LEROBOT_HOME="$release_check/hf/lerobot"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
printf 'Distribution verification workspace: %s\n' "$release_check"
cd "$release_root"
uv export --locked --extra dev --no-emit-project --no-hashes --format requirements-txt > "$release_check/constraints.txt"
release_wheels=("$release_root"/dist/lerobot_align-*.whl)
release_sdists=("$release_root"/dist/lerobot_align-*.tar.gz)
[[ ${#release_wheels[@]} == 1 && -f ${release_wheels[0]} ]]
[[ ${#release_sdists[@]} == 1 && -f ${release_sdists[0]} ]]
uv venv --python 3.12 "$release_check/wheel-env"
uv pip install --python "$release_check/wheel-env/bin/python" --constraint "$release_check/constraints.txt" "${release_wheels[0]}[dev]"
mkdir "$release_check/wheel-tests"
cp -R tests evaluation scripts pyproject.toml "$release_check/wheel-tests/"
cd "$release_check/wheel-tests"
"$release_check/wheel-env/bin/python" -c 'import lerobot_align; assert "wheel-env" in lerobot_align.__file__, lerobot_align.__file__'
"$release_check/wheel-env/bin/python" -m pytest -q tests evaluation -p no:cacheprovider
"$release_check/wheel-env/bin/python" -m tests.run_e2e_smoke
"$release_check/wheel-env/bin/python" -m tests.run_dependency_boundaries
for entry in lerobot-align lerobot-align-recover lerobot-align-eval lerobot-align-eval-batch lerobot-align-fit lerobot-motion-viz; do
    "$release_check/wheel-env/bin/$entry" --help > /dev/null
done
mkdir "$release_check/source"
tar -xzf "${release_sdists[0]}" -C "$release_check/source"
release_sources=("$release_check"/source/lerobot_align-*)
cd "${release_sources[0]}"
uv venv --python 3.12 "$release_check/sdist-env"
uv pip install --python "$release_check/sdist-env/bin/python" --constraint "$release_check/constraints.txt" '.[dev]'
"$release_check/sdist-env/bin/python" -m pytest -q tests evaluation -p no:cacheprovider
"$release_check/sdist-env/bin/ruff" check src tests evaluation
"$release_check/sdist-env/bin/python" -m tests.run_e2e_smoke
"$release_check/sdist-env/bin/python" -m tests.run_dependency_boundaries
for script in scripts/*.sh evaluation/scripts/*.sh; do bash -n "$script"; done
printf 'Wheel and extracted sdist checks passed: %s\n' "$release_check"
