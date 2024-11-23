from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

app = Flask(__name__)

with open("model.pickle", "rb") as model_file:
    classifier = pickle.load(model_file)

def preprocess_text(sentence):
    tokens = word_tokenize(sentence.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    features = {word: True for word in tokens}
    return features

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sentence = data.get("review", "")

    features = preprocess_text(sentence)

    prediction = classifier.classify(features)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")
    app.run(host="0.0.0.0", port=5000)
