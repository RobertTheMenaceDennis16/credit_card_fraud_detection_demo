from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

models = {
    "svm": joblib.load("models/svm_model.pkl"),
    "CS-SVM": joblib.load("models/cssvm_model.pkl"),
    "random_forest": joblib.load("models/rf_model.pkl"),
    "gradient_boost": joblib.load("models/gb_model.pkl"),
}

cards = {
    "clean_card": np.random.normal(0,1,28).tolist(),
    "stolen_card": np.random.normal(3,2,28).tolist()
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    card = request.form["card"]
    model_name = request.form["model"]

    features = cards[card]

    model = models[model_name]

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    if prediction == 1:
        result = "Fraud Detected 🚨"
    else:
        result = "Transaction Approved ✅"

    return render_template(
        "result.html",
        result=result,
        probability=round(probability,3),
        model=model_name
    )

if __name__ == "__main__":
    app.run(debug=True)