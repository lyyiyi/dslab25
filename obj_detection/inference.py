import wandb
from ultralytics import YOLO
import requests
from PIL import Image, ImageFile
from io import BytesIO
import os
import wandb
import weave

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize Weave and wandb with the same project name
project_name = "yolo_training"
weave.init(project_name)

# run = wandb.init(project=project_name)
# artifact = run.use_artifact('owendu-eth-z-rich-org/wandb-registry-model/yolov9_chart:v1', type='model')
# artifact_dir = artifact.download()

run = wandb.init()
# artifact = run.use_artifact('owendu-eth-z-rich-org/wandb-registry-model/yolov9_chart:v3', type='model')
# artifact_dir = artifact.download()

# # Define the path to the downloaded model
# model_path = os.path.join(artifact_dir, "yolov9c_finetunedv2.pt")
model_path = rf"/home/odu/chart_annotations/yolo_training/yolov9_finetune13/weights/best.pt"

# Load the pretrained YOLOv9 model
model = YOLO(model_path)

# Function to run inference on a single image
@weave.op
def run_inference(image: Image, ind) -> dict:
    try:
        # Save the image locally for prediction
        local_image_path = f'/home/odu/chart_annotations/data/test/temp_image{ind}.png'
        image.save(local_image_path)


        # Run the YOLO model on the image with adjusted NMS threshold
        results = model.predict(local_image_path, conf=0.7, iou=0.2)

        # Draw bounding boxes on the image and save the result
        results[0].save(local_image_path)
        result_image = Image.open(local_image_path)

        # Extract predictions
        predictions = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = results[0].names[class_id]
            confidence = box.conf.item()
            coordinates = box.xyxy.tolist()
            predictions.append({
                'class': class_name,
                'confidence': confidence,
                'coordinates': coordinates
            })


        # Prepare the results
        result_data = {
            'result_image': result_image,
            'predictions': predictions
        }


        return result_data
    except Exception as e:
        return {'error': str(e)}


# Download the image from the URL
ind = 0
test_img_dir = rf"/home/odu/chart_annotations/data/test_images"
for file in os.listdir(test_img_dir):
    file_path = os.path.join(test_img_dir, file)
    image = Image.open(file_path)

    # Run inference using the downloaded image
    inference_result = run_inference(image, ind)
    ind += 1
    print(inference_result)

