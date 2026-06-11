#!/bin/bash
# Reproduce the paper's fitness-convergence figure on a SLURM GPU node.
#
# Submit from the repository root:
#     sbatch scripts/run_paper_figures.slurm
#
# Edit the #SBATCH lines below to match your cluster (partition name is the most
# likely thing you need to change). Output figure lands in figures/.
#
#SBATCH --job-name=dts-figures
#SBATCH --partition=gpu              # <-- CHANGE to your GPU partition (e.g. gpu, main, rtx)
#SBATCH --gres=gpu:1                 # one GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00              # demo on a GPU is well under this; generous buffer
#SBATCH --output=slurm-dts-figures-%j.out

set -euo pipefail

# ----- run scale (change here if you want a different scale) -----
MODE=demo        # demo | paper | quick  (demo = GC 2000, SC 1000, TSP 1000)
RUNS=3

# ----- go to the repo root (the dir you ran `sbatch` from) -----
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Working dir: $(pwd)"

# ----- Python venv -----
# NOTE: if your cluster needs a module first, uncomment and adjust, e.g.:
#   module load python/3.11 cuda/12.1
if [ ! -d .venv ]; then
    echo "Creating .venv ..."
    python -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[plot]"   # core deps + matplotlib/pandas for plotting

# ----- sanity: a CUDA GPU must be visible -----
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available on this node — check --gres/--partition and the torch build"
print("CUDA OK:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PY

# ----- fresh run: clear cached runs so config changes actually take effect -----
# (make_paper_figures skips runs whose JSON already exists; wipe them for a clean re-run)
echo "Clearing previous cached runs so the current config is used ..."
rm -rf runs/paper_figures figures/reproduction_fitness_all_domains.png figures/reproduction_fitness_all_domains.pdf

# ----- run: DTS vs Tournament on the paper's representative instances, on GPU -----
python make_paper_figures.py --mode "$MODE" --runs "$RUNS" --device cuda \
    --output runs/paper_figures --figdir figures

echo "Done. Figure: figures/reproduction_fitness_all_domains.png (and .pdf)"
echo "Per-run JSONs (resumable): runs/paper_figures/<domain>/<instance>/<selection>/run_*.json"
