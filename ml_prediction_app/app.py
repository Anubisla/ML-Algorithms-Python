# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        hours = float(request.form["hours"])
        sleep = float(request.form["sleep"])

        # Prediction
        features = np.array([[hours, sleep]])
        prediction = model.predict(features)[0]

        return render_template("index.html", result=round(prediction, 2))

    except:
        return render_template("index.html", result="Error in input")


if __name__ == "__main__":
    app.run(debug=True)
