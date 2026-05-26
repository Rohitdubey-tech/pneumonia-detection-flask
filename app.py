import os
import cv2
import base64
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image

# ================= APP SETUP =================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_resnet18.pth")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

CLASSES = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= X-RAY VALIDATION =================
def is_xray(img):
    """
    Heuristic validation to reject non–X-ray images
    """
    if img is None:
        return False

    # Mostly grayscale check
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        if np.mean(np.abs(b - g)) > 10 or np.mean(np.abs(g - r)) > 10:
            return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast check
    contrast = gray.std()
    if contrast < 15 or contrast > 80:
        return False

    # Brightness check
    brightness = gray.mean()
    if brightness > 160:
        return False

    return True

# ================= PREPROCESS =================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

# ================= PREDICTION =================
def predict(img):
    pil_img = Image.fromarray(img).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    normal_p = probs[0].item() * 100
    pneumonia_p = probs[1].item() * 100

    confidence = max(normal_p, pneumonia_p)
    label = "PNEUMONIA" if pneumonia_p > normal_p else "NORMAL"

    return label, round(confidence, 2)

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/camera")
def camera():
    return render_template("camera.html")

# -------- UPLOAD MODE --------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return render_template("index.html", error="No file uploaded")

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    img = cv2.imread(path)
    if not is_xray(img):
        return render_template(
            "index.html",
            error="Please upload a valid chest X-ray"
        )

    scanned = preprocess(img)
    label, confidence = predict(scanned)

    if confidence < 70:
        return render_template(
            "index.html",
            error="Image unclear or not a valid chest X-ray"
        )

    scanned_name = "scanned_" + file.filename
    scanned_path = os.path.join(UPLOAD_DIR, scanned_name)
    cv2.imwrite(scanned_path, scanned)

    return render_template(
        "index.html",
        image_path=f"static/uploads/{file.filename}",
        scanned_path=f"static/uploads/{scanned_name}",
        prediction=label,
        confidence=confidence
    )

# -------- CAMERA MODE --------
@app.route("/scan_camera", methods=["POST"])
def scan_camera():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image captured"})

    img_bytes = base64.b64decode(data.split(",")[1])
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if not is_xray(img):
        return jsonify({"error": "Please capture a valid chest X-ray"})

    scanned = preprocess(img)
    label, confidence = predict(scanned)

    if confidence < 70:
        return jsonify({"error": "Image unclear or not a valid chest X-ray"})

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)
