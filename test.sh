#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --account=dslab
#SBATCH --mincpus=1
#SBATCH --gpus=1

# Default values
CONDA_ENV_NAME="dsl"
PYTHON_SCRIPT="dslab25/obj_detection/dino/inference.py"
DATA_DIR="/work/courses/dslab/team14"

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
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: sbatch $0 [--env <env>] [--script <script_path>] [--data-dir <data_path>]"
      exit 1
      ;;
  esac
done

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

echo "Running on host: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Using Conda env: $CONDA_ENV_NAME"
echo "Running script: $PYTHON_SCRIPT"
echo "DATA_DIR: $DATA_DIR"

set -o errexit

VIDEO_DIR="$DATA_DIR/videos/input_video"
for label_file in "$VIDEO_DIR"/*_labels_5fps.txt; do
    label_filename=$(basename "$label_file")

    if [[ "$label_filename" =~ ([0-9]+)_run([0-9]+)_labels_5fps.txt ]]; then
        subject="${BASH_REMATCH[1]}"
        run="${BASH_REMATCH[2]}"
        video_file="${subject}_run${run}_cam_2_5fps.mp4"
        video_path="$VIDEO_DIR/$video_file"
        output_file="${subject}_run${run}.out"

        if [[ -f "$video_path" ]]; then
            echo "Processing subject $subject run $run (output -> $output_file)"

            # Run Python script, redirect output, and handle failure gracefully
            if ! srun python3 "$PYTHON_SCRIPT" \
                --labels_file "$label_file" \
                --video_path "$video_path" \
                --subject "$subject" \
                --run "$run" > "$output_file" 2>&1; then
                echo "Python script failed for subject $subject run $run. Skipping..."
            fi
        else
            echo "Missing video for subject $subject run $run"
        fi
    else
        echo "Could not parse subject/run from $label_filename"
    fi
done



echo "Finished at: $(date)"
exit 0
