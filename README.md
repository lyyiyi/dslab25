# Data Science Lab FS 2025

Authors: Yi-Yi Ly, Georg Ye, Owen Du

## Repo Structure

```
.
├── src
└── data/
    ├── metadata.csv
    ├── Vacuum_pump/
    ├── Vacuum_pump_step_file/
    ├── class_sequence.txt
    └── classes.txt
```

## Merge Requests

We follow the 4-eye-principle before merging any branches.

feature branch -> dev -> main

## Commits

We follow Conventional Commits to make all commit messages coherent.

## Guide to use of student-cluster

Each student can use one GPU for up to 100 hours, and no more than one GPU at the same time. You have 20 GB of space in your home directory and extra space for data in `/work/courses/dsl/team14`
You can always access the cluster from an ETH wifi. To access the cluster from remote, you need a VPN: https://www.isg.inf.ethz.ch/Main/ServicesNetworkVPN.
The cluster uses Slurm and a general guide to the cluster can be found at: https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster.

You can follow the steps in this message to quickly login the first time and create your conda environment. For more details visit the link above.Login to cluster:
To login to the student cluster, open your powershell and connect through ssh to one of the 2 login nodes of the student cluster:

`ssh <username>@student-cluster{?2}.inf.ethz.ch` (either include 2 or not)

If you want to use Jupyter you can login from this link: https://student-jupyter.inf.ethz.ch.Using

### conda:

Navigate to the directory that already contains conda:

`cd /cluster/data/miniconda`

You can check the available versions with ls and install the version that you prefer, for example:

`./Miniconda3-py310_23.10.0-1-Linux-x86_64.sh`

Accept the terms and complete the installation. Navigate to your home directory using cd . Now initialize conda:

`~/miniconda3/bin/conda init`

Close and re-open the shell. After logging in again, you should see written (base) at the beginning of the line, and the conda commands should work. You can test it with conda `--version`.Create now a conda environment using the python version that you prefer and activate it:

`conda create -n your_env_name python==3.x -y`

`conda activate your_env_name`

If you want to always open your environment when connecting to the cluster, run:

`nano ~/.bashrc`

And add this after the initialization of conda:

`conda activate your_env_name`

When the environment is activated, you can install packages with pip or conda, e.g. pip install transformers .Running jobs:
A good guide on how to run a job with slurm on the student cluster is provided at https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs.
Keep in mind that the {course tag} for this course is dsl. (or dslab)

### jobs

```
You have the following amount of hours for courses left:

  100 of 100 GPU hours left for dslab [run time 60 minutes]
  100 of 100 GPU hours left for dslab_jobs [max run time 24 hours]

Your home has 19816MB free space of 20000MB total.
```

run.sh contains training script \
you need to change conda activate <env> to your env name\
add your WANDB API key (don't push it, add script file to .gitignore)\
run training script using

`sbatch run.sh`

## Blender

### Textures given to us by the challenger giver:

They are wrong lol.

```json
{
    # Gehaeuse_1_1_bottom
    "case_bottom": {
      "color": [0.59, 0.51, 0.43, 1.0] ,
      "roughness": 0.65,
      "metallic": 1.0
    },
    # rotor/roter
    "axel": {
      "color": [0.30, 0.20, 0.15, 1.0],
      "roughness": 0.9,
      "metallic": 0.2
    },
    # Gehaeuse_teil_2_1
    "case_upper": {
      "color": [0.59, 0.51, 0.43, 1.0] ,
      "roughness": 0.65,
      "metallic": 1.0
    },
    # abstandplatte 1
    "diamond": {
      "color": [0.30, 0.20, 0.15, 1.0],
      "roughness": 0.65,
      "metallic": 0.9
    },
    # Abdeckplatte
    "cover_top": {
      "color": [0.01, 0.01, 0.01, 1],
      "roughness": 0.5,
      "metallic": 0
    }
  }
```

### Install the blender add on (no longer needed):

https://blendermarket.com/products/physics-dropper

### Run the blender script

`/Applications/Blender.app/Contents/MacOS/Blender --python blender_new.py`

### Generate annotations

This is old, dont use this anymore. If you need to create annotations from scratch write a new script

`python3 annotation.py --root /Users/georgye/Documents/repos/ethz/dslab25/training vacuum_pump/generated/output/stages --output-dir ./anno`

### Useful commands

Zip folder without .git and venv:
`zip -r dslab25.zip dslab25 -x "dslab25/.git/*" "dslab25/venv/*" "dslab25/obj_detection/dino/yolo_runs/*"`

Send data on runpod:
`runpodctl send data.txt`

### Augment images

We do the following

1. Rotation (20 degrees increements)
2. Brightness
3. Obscure

First go to `obj_detection/preproccessing/augment.ipynb` and run the Rotate (Images) cell.
Then go to roboflow and annotate them manually, then put them in `obj_detection/preproccessing/stage_0/labels` and run the rest of the

## Running Dino

### Preprocessing

You need to create a `coco.json` first. Run all cells of `obj_detection/dino/coco.ipynb` first.

### Training

Run this command (replace n with whatever how many gpus you have)
`torchrun --nproc_per_node=n obj_detection/dino/train.py`

### Testing model weights

Note you need to model weights in `obj_detection/dino/final_model`.
Run:
`python3 obj_detection/dino/eval.py`

# TODO:

- augment with scale and shear
- blender one with and without random objects
- randomize objects in each render in blender
