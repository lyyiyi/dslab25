from ultralytics import YOLO
import wandb

# Load a pretrained YOLOv9 model
model = YOLO("yolov9c.pt")
model_path = "yolov9c_finetunedv2.pt"
# Train the model on your datasetfine
results = model.train(
    data="./data.yaml",
    epochs=100,
    imgsz=640,
    entity="dslab25",
    project="yolo",  # wandb project name
    name="yolov9_finetune",     # wandb run name
    device="0"
)

# Save the fine-tuned model
model.save(model_path)

wandb.init("yolo")
wandb.log_model(model_path)
wandb.finish()
