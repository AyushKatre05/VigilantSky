import os
import datetime
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load YOLOv5 model for drone detection
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ayush/OneDrive/Desktop/bird/Drone-Detection-Model/runs/detect/train13/weights/best.pt', source='github')

# Create directories to store images
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

@app.route("/")
def home():
    return "Drone Detection Backend Running!"

def process_image(image_path):
    """Process an image using YOLOv5 for drone detection."""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image"}

    img = Image.fromarray(image[..., ::-1])  # Convert OpenCV BGR to RGB
    results = model(img, size=640)

    detected_objects = []
    
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            text = f"Drone {conf * 100:.2f}%"
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            detected_objects.append({
                "class": "Drone",
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

    processed_filename = "processed_" + os.path.basename(image_path)
    processed_path = os.path.join("processed", processed_filename)
    cv2.imwrite(processed_path, image)

    # Fix result_image URL to be relative
    return {
        "detections": detected_objects,
        "result_image": f"/processed/{processed_filename}"  # This ensures frontend adds the base URL correctly
    }

@app.route("/detect/drone", methods=["POST"])
def detect_drone():
    """Detect drones in an uploaded image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = datetime.datetime.now().strftime("drone_%Y-%m-%d_%H-%M-%S.jpg")
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    results = process_image(filepath)
    return jsonify(results)

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("processed", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
