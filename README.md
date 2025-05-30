# Data Science Lab FS 2025

Authors: Yi-Yi Ly, Georg Ye, Owen Du

![Preview](intro.gif)

Rejected frames are shown in red, accepted frames in green. Only green frames are classified.

## Repo Structure

```
.

```

## Downsample the videos

Run `dslab25/obj_detection/preprocessing/downsample.ipynb` to downsample the original videos to 5fps. Hereby, change `DATA` path to the path of the original folder, which contains the `.avi` files.

## Installing (requirements, SAM2, blender)

```
sudo apt -y update
sudo apt -y install pkg-config libcairo2-dev libgirepository1.0-dev python3-dev python3-setuptools zip unzip
pip install --upgrade pip setuptools wheel
pip install --no-build-isolation iopath
```

pip install -r requirements.txt

## Run whole pipeline (extract seedfram -> frame rej -> inference)

Run `dslab25/obj_detection/dino/pipeline.py` to:

1. obtain the seedframe using cosimilarity search on DINO features on bounding boxes of YOLO and reference object
2. perform SAM tracking starting seed frame that was found
3. frame rejection based on IoU and scale difference
4. inference using DINO

Results can be analyzed in `dslab25/obj_detection/dino/final_results.ipynb`.

To **test** the frame rejection framework and obtain the confusion matrix for a video, run (from home directory):

```
./obj_detection/dino/test_frame_rejection.py
```

## Analysis of Results

# The final results, statistics can be performed in:

```
./obj_detection/dino/final_results.ipynb
```

The masks and contours can be analyzed using:

```
./obj_detection/dino/play_masks_analysis.ipynb
```

## Blender

### One can adjust the material of the objects, the one which was given to us was:

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

`/Applications/Blender.app/Contents/MacOS/Blender --python blender_single.py`
or
`/Applications/Blender.app/Contents/MacOS/Blender --python blender_messy_table.py`

## Generate annotations

This is old, dont use this anymore. If you need to create annotations from scratch write a new script

`python3 annotation.py --root /Users/georgye/Documents/repos/ethz/dslab25/training vacuum_pump/generated/output/stages --output-dir ./anno`

## Preprocessing

### Augment images

We do the following

1. Rotation (20 degrees increements)
2. SAM features
3. Tint patches
4. Brightness
5. Obscure
6. Scale
7. Translate

This can quite some time, and be be pretty big (around 100GB (I think)).

First go to `obj_detection/preproccessing/augment.ipynb` and run the Rotate (Images) cell.
Then go to roboflow and annotate them manually, then put them in `obj_detection/preproccessing/stage_0/labels` and run the rest of the cells (You dont need to do it now because I already did it).

## Training YOLO

run all cells of `obj_detection/dino/yolo.ipynb`

## Training Dino

run all cells of `obj_detection/dino/dino.ipynb`

## Inference

run all cells of `obj_detection/dino/inference.ipynb`

## Qwen 2.5 VL (experimental)

### Testing model weights

Put the yolo model weights in `obj_detection/dino/yolo_runs/yolov12_boundingbox2/weights/best.pt`
Put the dino model weights in `obj_detection/dino/dinov2_finetune/base/final_model/`

One can download the weights we got here for now (will be deleted at some point): https://drive.google.com/drive/folders/1jG1oAAIPkC0HQ8rD882kmf3sLeGSawBg

## Useful commands

Zip folder without .git and venv:
`zip -r dslab25.zip dslab25 -x "dslab25/.git/*" "dslab25/venv/*" "dslab25/obj_detection/dino/yolo_runs/*"`

Send data on runpod:
`runpodctl send data.txt`
