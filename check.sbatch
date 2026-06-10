#!/bin/bash
# Quick sanity check on a SLURM cluster: runs DTS for a few generations on one
# instance of each problem, to confirm the install works end-to-end.
# This is NOT the paper protocol — just a smoke test.
#
# Usage (after the one-time login-node setup below):
#   sbatch check.sbatch
#   tail -f dts-check-<jobid>.out
#
# ----- ADJUST these to your cluster (ask your admin / check `sinfo`) -----
#SBATCH --job-name=dts-check
#SBATCH --output=dts-check-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1            # request 1 GPU; comment out this line for CPU-only
## #SBATCH --partition=<your-partition>
## #SBATCH --account=<your-account>
# -------------------------------------------------------------------------

set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# auto-pick device: cuda if a GPU is visible, else cpu
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    DEV=cuda
else
    DEV=cpu
fi
echo "Python: $(python --version)   |   device: $DEV"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo "### Set Cover (scp41, 50 gens) ###"
python -m deep_tournament_selection.experiments.set_cover --instance scp41.txt --generations 50 --device $DEV

echo "### Graph Coloring (queen8_12, 50 gens) ###"
python -m deep_tournament_selection.experiments.graph_coloring --instance queen8_12.col.txt --generations 50 --device $DEV

echo "### TSP (att48, 50 gens) ###"
python -m deep_tournament_selection.experiments.tsp --instance att48.tsp --generations 50 --device $DEV

echo "ALL CHECKS DONE"
