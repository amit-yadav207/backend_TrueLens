import os
import platform
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pytesseract
from PIL import Image
import neattext.functions as nfx
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Detect the OS and set the Tesseract path accordingly
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Load dataset and preprocess
df = pd.read_csv("Constraints1.csv")
df.drop("id", axis=1, inplace=True)
df["label"] = df["label"].map({"real": 0, "fake": 1})  # Convert labels to numerical values
df["Clean_Text"] = df["tweet"].apply(nfx.remove_userhandles).apply(nfx.remove_stopwords)
df.drop("tweet", axis=1, inplace=True)

# Split dataset into training and testing sets
X = df["Clean_Text"]
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Model Pipelines
models = {
    "Logistic Regression": Pipeline([("cv", CountVectorizer()), ("lr", LogisticRegression())]),
    "Decision Tree": Pipeline([("cv", CountVectorizer()), ("lr", DecisionTreeClassifier(max_depth=6, random_state=42))]),
    "KNN": Pipeline([("cv", CountVectorizer()), ("lr", KNeighborsClassifier(n_neighbors=15))]),
    "Random Forest": Pipeline([("cv", CountVectorizer()), ("lr", RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42))]),
    "AdaBoost": Pipeline([("cv", CountVectorizer()), ("lr", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
               n_estimators=200, learning_rate=0.5, random_state=42))])
}

# Train models
for name, model in models.items():
    model.fit(x_train, y_train)

# Function to get averaged predictions from multiple models
def predict_tweets(text):
    results = {name: model.predict_proba([text])[0] for name, model in models.items()}
    
    avg_true_prob = sum(results[name][0] for name in results) / len(results)
    avg_fake_prob = sum(results[name][1] for name in results) / len(results)

    return {
        "true_percentage": round(avg_true_prob * 100, 2),
        "false_percentage": round(avg_fake_prob * 100, 2)
    }

@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    response = predict_tweets(text)
    return jsonify(response)

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = Image.open(request.files["image"])
    extracted_text = pytesseract.image_to_string(image, lang='eng').strip()

    if not extracted_text:
        return jsonify({"error": "No text could be extracted from the image"}), 400

    response = predict_tweets(extracted_text)
    response["text"] = extracted_text  # Include extracted text in response
    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's provided port
    app.run(host="0.0.0.0", port=port, debug=False)  # Bind to all interfaces
    # app.run(debug=True)
