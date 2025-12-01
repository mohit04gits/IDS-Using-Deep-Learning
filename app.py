import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle

app = Flask(__name__)
CORS(app)

# Load model + scaler
model = tf.keras.models.load_model("IDS_ANN_Model.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        if features.shape[1] != 7:
            return jsonify({"error": "Expected 7 features"}), 400

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0][0]
        result = 1 if prediction >= 0.5 else 0

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/", methods=["GET"])
def home():
    return "IDS Backend Running Successfully!"


if __name__ == "__main__":
    app.run(debug=True)
