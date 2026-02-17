from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os
import time

app = Flask(__name__)

# ===== MODEL LOAD =====
MODEL_PATH = "model/plant_disease_cnn_final.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

IMG_SIZE = 224   # model ke input size ke according

# ===== PREDICT ROUTE =====
@app.route('/predict', methods=['POST'])
def predict():

    # 1️⃣ check image received
    if request.data is None or len(request.data) == 0:
        return "No image received", 400

    # 2️⃣ decode image
    img_array = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Image decode failed", 400

    # 3️⃣ save received image (optional but useful)
    if not os.path.exists("received_images"):
        os.mkdir("received_images")

    cv2.imwrite(f"received_images/img_{int(time.time())}.jpg", img)

    # 4️⃣ preprocess image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 5️⃣ prediction
    prediction = model.predict(img)[0][0]

    # 6️⃣ result logic (IMPORTANT PART)
    if prediction >= 0.3:
        result = "Infected"
    else:
        result = "Healthy"

    print("✅ Prediction:", prediction, "=>", result)

    # 7️⃣ Arduino-friendly response
    return result, 200


# ===== FLASK START =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)