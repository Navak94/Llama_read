#!/bin/bash
#SBATCH -p long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -C A100|H100|H200
#SBATCH --job-name=LlamaRead
#SBATCH -t 164:00:00
#SBATCH --output=./logs/%x-%j.out

# Fail fast and be strict
# set -euo pipefail

# # Ensure logs directory exists
# mkdir -p logs

# module load python/3.10.17

set -euo pipefail

cd /home/nthindman/scratch/Llama_read
source venv/bin/activate

# # Your other deps
# uv pip install numpy==2.0 singd==0.0.5 matplotlib tqdm pytorch_msssim lpips torch torchaudio torchvision

# # Some clusters set this; make sure it's clean
# unset LD_PRELOAD || true

# # NCCL / networking environment (adjust if your fabric differs)
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=^mlx5_2:1

# # Show primary node IP (optional debug)
# # shellcheck disable=SC2207
# nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname --ip-address)
# echo "Rank 0: Node ${nodes[0]}, IP: ${head_node_ip}"

# ----- Run the specific Python file passed via --export=FILE=... -----
if [[ -z "${FILE:-}" ]]; then
  echo "ERROR: FILE env var not set. Submit with: sbatch --export=ALL,FILE=<your.py> run_one.slurm"
  exit 2
fi

echo ">>> Running: ${FILE}"
python "${FILE}"
