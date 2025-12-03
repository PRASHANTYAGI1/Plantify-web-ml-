from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import os
import uvicorn
import tempfile
import json

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ PATHS ------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_MODEL_PATH = os.path.join(BASE_DIR, "crop_recommendation_ann.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.save")
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "potato_model.h5")

# ------------ LOAD MODELS ------------

# Crop model
crop_model = load_model(CROP_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Potato model (optional)
potato_model = None
if os.path.exists(POTATO_MODEL_PATH):
    potato_model = load_model(POTATO_MODEL_PATH)


# ------------ CROP PREDICTION ------------
@app.post("/predict_crop")
async def predict_crop(payload: dict):
    values = payload["values"]   # list of 7 numbers

    arr = np.array([values])
    scaled = scaler.transform(arr)
    probs = crop_model.predict(scaled)[0]

    prob_list = {label: float(probs[i]) for i, label in enumerate(le.classes_)}
    sorted_probs = dict(sorted(prob_list.items(), key=lambda x: x[1], reverse=True))
    pred_crop = list(sorted_probs.keys())[0]

    return {
        "predictedCrop": pred_crop,
        "probabilities": sorted_probs
    }


# ------------ POTATO DISEASE PREDICTION ------------
@app.post("/predict_potato")
async def predict_potato(image: UploadFile = File(...)):

    if potato_model is None:
        return {"error": "Potato model not available"}

    # Save temporary file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, image.filename)

    with open(file_path, "wb") as f:
        f.write(await image.read())

    # Run prediction
    img = Image.open(file_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, 0)
    probs = potato_model.predict(arr)[0]

    classes = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    prob_list = {classes[i]: float(probs[i]) for i in range(len(classes))}
    pred_class = classes[int(np.argmax(probs))]

    return {
        "predictedClass": pred_class,
        "probabilities": prob_list
    }


# ------------ START SERVER ------------
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

