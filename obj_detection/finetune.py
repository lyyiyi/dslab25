from ultralytics import YOLO
import wandb

# Load a pretrained YOLOv9 model
model = YOLO("yolov9c.pt")
model_path = "obj_detection/yolov9c_finetunedv2.pt"
# Train the model on your datasetfine
results = model.train(
    data="/home/owendu/dslab25/obj_detection/data.yaml",
    epochs=100,
    imgsz=512,
    project="yolo",  # wandb project name
    name="yolov9_finetune",     # wandb run name
)

# Save the fine-tuned model
model.save(model_path)

wandb.init("yolo")
wandb.log_model(model_path)
wandb.finish()
