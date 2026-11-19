#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root=$PWD
run_dir="$root/data/seed-evolution/2026-11-11-large-v1"
command=${1:-run}
if (( $# > 0 )); then shift; fi

if [[ "$command" == status ]]; then
    cat "$run_dir/progress.json"
    exit 0
fi
if [[ "$command" != run && "$command" != check ]]; then
    echo 'Usage: bash run.sh [run [--workers N] | check | status]' >&2
    exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
    cpu_python=$PYTHON
elif command -v python3.13 >/dev/null 2>&1; then
    cpu_python=python3.13
else
    cpu_python=python3
fi
"$cpu_python" -c 'import platform, sys; assert sys.platform == "linux" and platform.machine() == "x86_64" and sys.version_info[:2] == (3, 13), "This wheelhouse requires Linux x86_64 and Python 3.13"'

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
runtime="$root/.cpu-runtime"
wheels="$root/wheelhouse"
pip_wheel="$wheels/pip-26.1.1-py3-none-any.whl"

verify_runtime() {
    PYTHONPATH="$1" "$cpu_python" -c 'import numpy, torch; assert torch.__version__ == "2.11.0+cpu" and torch.version.cuda is None and numpy.__version__ == "2.5.4"; assert (torch.tensor([2.0]) ** 2).item() == 4.0'
}

exec 9> "$root/.cpu-setup.lock"
if command -v flock >/dev/null 2>&1; then flock 9; fi
if [[ ! -f "$runtime/.ready" ]] || ! verify_runtime "$runtime"; then
    if [[ ! -f "$pip_wheel" || ! -f "$wheels/SHA256SUMS" ]]; then
        echo 'The bundled Linux CPU wheels are missing. Extract the full archive.' >&2
        exit 1
    fi
    (cd "$wheels" && sha256sum -c SHA256SUMS)
    staging=$(mktemp -d "$root/.cpu-runtime.install.XXXXXX")
    PYTHONPATH="$pip_wheel" "$cpu_python" -m pip --isolated install \
        --no-index --find-links "$wheels" --only-binary=:all: --ignore-installed \
        --target "$staging" 'torch==2.11.0+cpu' 'numpy==2.5.4'
    verify_runtime "$staging"
    touch "$staging/.ready"
    if [[ -e "$runtime" ]]; then
        backup=$(mktemp -d "$root/.cpu-runtime.previous.XXXXXX")
        mv -- "$runtime" "$backup/runtime"
    fi
    mv -- "$staging" "$runtime"
fi
exec 9>&-

export PYTHONPATH="$root:$runtime"
exec "$cpu_python" -u -m experiments.seed_evolution.cloud_cpu "$command" "$@"
