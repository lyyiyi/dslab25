from ultralytics import YOLO
import wandb

# Load a pretrained YOLOv9 model
model = YOLO("yolo11n.pt")
model_path = "obj_detection/yolov9c_finetunedv2.pt"
# Train the model on your datasetfine
results = model.train(
    data="/Users/owen/Projects/dslab25/obj_detection/data.yaml",
    epochs=100,
    imgsz=512,
    hyp="/Users/owen/Projects/dslab25/obj_detection/hyp.yaml",
    project="yolo",  # wandb project name
    name="yolov9_finetune",     # wandb run name
    device='mps'
)

# Save the fine-tuned model
model.save(model_path)

wandb.init("yolo")
wandb.log_model(model_path)
wandb.finish()
