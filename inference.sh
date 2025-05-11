#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --account=dslab
#SBATCH --output=%j.out
#SBATCH --mincpus=1
#SBATCH --gpus=1

# Default values
CONDA_ENV_NAME="dsl"
PYTHON_SCRIPT="dslab25/obj_detection/dino/inference.py"

# export WANDB_API_KEY=6c3d2b57c8093f64a7cbb9710592fad1e14f8fa6

# Parse named parameters
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --script)
      PYTHON_SCRIPT="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: sbatch $0 [--env <env>] [--script <script_path>]"
      exit 1
      ;;
  esac
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
/bin/echo Using Conda env: $CONDA_ENV_NAME
/bin/echo Running script: $PYTHON_SCRIPT

set -o errexit

srun python3 "$PYTHON_SCRIPT"

echo finished at: `date`
exit 0
