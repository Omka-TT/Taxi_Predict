from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__, static_folder='static')

model = joblib.load("taxi_fare_model.pkl")


@app.route("/")
def home():
    metrics = {}
    with open("metrics.txt", "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(": ", 1)
                metrics[key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("²", "2")] = value
    sample_data = pd.read_csv('train.csv').head(10).to_dict('records')
    return render_template("index.html", **metrics, sample_data=sample_data)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Получаем данные из формы
        distance_traveled = float(request.form["distance_traveled"])
        trip_duration = float(request.form["trip_duration"])
        fare = float(request.form["fare"])

        features = np.array([[distance_traveled, trip_duration, fare]])

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)

