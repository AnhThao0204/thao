from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# =====================
# FIX CHO MÁY AMD (CPU)
# =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Tắt GPU (tránh TF tìm CUDA)

tf.config.set_visible_devices([], 'GPU')  # Ép TensorFlow chạy CPU

# =====================
# FLASK APP
# =====================
app = Flask(__name__)

# Load model
model = load_model(r'C:\Users\nguye\Downloads\demoo\model.h5')

IMG_WIDTH, IMG_HEIGHT = 128, 128

# =====================
# PREPROCESS
# =====================
def preprocess_image(image, target_size):
    image = image.convert("RGB")          # BẮT BUỘC tránh lỗi ảnh RGBA
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================
# PREDICT
# =====================
def predict_image(image):
    img = preprocess_image(image, (IMG_WIDTH, IMG_HEIGHT))

    prediction = model.predict(img, verbose=0)[0][0]

    dog_probability = float(prediction * 100)
    cat_probability = float((1 - prediction) * 100)

    return cat_probability, dog_probability

# =====================
# ROUTES
# =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files[]' not in request.files:
        return redirect(url_for('index'))

    images = []
    for file in request.files.getlist('files[]'):
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img = Image.open(BytesIO(img_bytes))
        images.append((img_base64, img))

    predictions = [(img_str, predict_image(img)) for img_str, img in images]

    return render_template('index.html', predictions=predictions)

# =====================
# API (POSTMAN)
# =====================
@app.route('/predict-api', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    try:
        img = Image.open(request.files['image'].stream)
    except:
        return jsonify({"error": "Invalid image"}), 400

    cat_probability, dog_probability = predict_image(img)

    return jsonify({
        "cat_probability": round(cat_probability, 2),
        "dog_probability": round(dog_probability, 2)
    })

# =====================
# MAIN
# =====================
if __name__ == '__main__':
    app.run(debug=True)
