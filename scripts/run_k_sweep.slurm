#!/bin/bash
# Tournament-size (k) sweep across all problems, on a SLURM GPU node.
#
#     sbatch scripts/run_k_sweep.slurm
#
#SBATCH --job-name=dts-ksweep
#SBATCH --partition=gpu               # <-- your GPU partition
#SBATCH --gres=gpu:rtx_3090:1         # <-- PIN the GPU type. The installed torch only
                                      #     has kernels for some architectures; landing on
                                      #     an unsupported GPU gives "no kernel image is
                                      #     available for execution on the device". RTX 3090
                                      #     is known-good here; change if you pin another.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00               # the full sweep is long; resumable, so re-submit to continue
#SBATCH --output=slurm-dts-ksweep-%j.out

set -euo pipefail

# ----- knobs -----
K_VALUES="3 5 10 20"
SELECTION=dts        # dts | tournament | both
RUNS=15

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Working dir: $(pwd)  | node: $(hostname)"

# ----- Python venv -----
# (uncomment if your cluster needs a module first)
#   module load python/3.11 cuda/12.1
if [ ! -d .venv ]; then
    python -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[plot]"

# ----- CUDA sanity: actually LAUNCH a kernel, so an incompatible GPU fails NOW
#       (not 3 hours into the sweep) with a clear message. -----
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — check --gres/--partition"
try:
    x = torch.randn(128, 128, device="cuda")
    _ = float((x @ x).sum())          # forces a real kernel launch
except RuntimeError as e:
    raise SystemExit(
        f"GPU kernel launch failed on {torch.cuda.get_device_name(0)}: {e}\n"
        "This torch build has no kernels for this GPU. Pin a supported GPU via\n"
        "#SBATCH --gres=gpu:<type>:1 (e.g. rtx_3090), or reinstall a matching torch."
    )
print("CUDA OK:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PY

# ----- the sweep -----
python run_k_sweep.py --k-values $K_VALUES --selection "$SELECTION" --runs "$RUNS" \
    --device cuda --output runs/k_sweep

echo "Done. Per-run JSONs under runs/k_sweep/ ; summary: runs/k_sweep/k_sweep_summary.csv"
