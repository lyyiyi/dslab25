#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --account=dslab
#SBATCH --output=%j.out
#SBATCH --mincpus=1
#SBATCH --gpus=1

source ~/miniconda3/etc/profile.d/conda.sh

conda activate env

export WANDB_API_KEY=6c3d2b57c8093f64a7cbb9710592fad1e14f8fa6

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

srun python3 obj_detection/finetune.py
echo finished at: `date`
exit 0;