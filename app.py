import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Define preprocessing function
def preprocess_text(text):
    import re
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d", "", text)  # Remove digits
    text = re.sub(r"\n", " ", text)  # Remove newlines
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from form
    news_text = request.form['news_text']

    # Preprocess and transform text
    processed_text = preprocess_text(news_text)
    vectorized_text = vectorizer.transform([processed_text])

    # Predict using model
    prediction = model.predict(vectorized_text)[0]

    # Convert prediction to readable text
    result = "Genuine News" if prediction == 1 else "Fake News"

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
