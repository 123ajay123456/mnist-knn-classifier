from flask import Flask, request, jsonify, render_template
import re
import base64
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load saved KNN and PCA
(knn, pca) = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Invert: if your canvas is white background with black ink
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    # Resize whole canvas to 28x28 directly
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Debug save
    cv2.imwrite('processed.png', img)

    # Flatten & normalize
    img_flat = img.flatten().astype(np.float32) / 255.0

    img_pca = pca.transform([img_flat])
    prediction = knn.predict(img_pca)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
