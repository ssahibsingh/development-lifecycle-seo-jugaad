import numpy as np
from flask import Flask, request, jsonify, has_request_context
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# Home Route
@app.route("/", methods=["GET", "POST"])
@cross_origin()
def home():
    if request.method == "GET":
        return jsonify({"status": "active", "message": "Teachable Machine Flask"}), 200


# Predict Route
@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "GET":
        return jsonify({"status": "active", "message": "Predict Actions"}), 200
    elif request.method == "POST":
        # Get uploaded image
        uploadedImage = request.files["image"]

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = load_model("./model/keras_Model.h5", compile=False)

        # Load the labels
        class_names = open("./model/labels.txt", "r").readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(uploadedImage).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return jsonify({
            "status": "active",
            "prediction": str(class_name[2:]),
            "confidence": float(confidence_score*100)
            }), 200

# Running the app
if __name__ == "__main__":
    app.run(debug=True)