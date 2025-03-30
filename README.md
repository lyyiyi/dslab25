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

### Blender add on

https://blendermarket.com/products/physics-dropper
/Applications/Blender.app/Contents/MacOS/Blender --python blender_new.py

{
"case_bottom": { # Gehaeuse_1_1_bottom
"color": [0.59, 0.51, 0.43, 1.0] , "roughness": 0.65, "metallic": 1.0
},
"axel": { # rotor/roter
"color": [0.30, 0.20, 0.15, 1.0] , "roughness": 0.9, "metallic": 0.2
},
"case_upper": { # Gehaeuse_teil_2_1
"color": [0.59, 0.51, 0.43, 1.0] , "roughness": 0.65, "metallic": 1.0
},
"diamond": { # abstandplatte 1
"color": [0.30, 0.20, 0.15, 1.0] , "roughness": 0.65, "metallic": 0.9
},
"cover_top": { Abdeckplatte
"color": [0.01, 0.01, 0.01, 1] , "roughness": 0.5, "metallic": 0
}
}

### Generate annotations

python3 annotation.py --root /Users/georgye/Documents/repos/ethz/dslab25/training/vacuum_pump/generated/output/stages --output-dir ./anno
