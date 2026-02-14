from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

app = Flask(__name__)

# ---------- Load Dataset Safely ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "bot.csv")

data = pd.read_csv(csv_path)

x = data['q']
y = data['a']

# ---------- Train Model ----------
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(x)

model = MultinomialNB()
model.fit(X_vec, y)


# ---------- Chatbot Function ----------
def chatbot_response(user_input):
    if not user_input:
        return "Please enter a valid message."

    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)
    return prediction[0]


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_bot_response():
    data = request.get_json()

    if not data or "msg" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_input = data["msg"]
    response = chatbot_response(user_input)

    return jsonify({"response": response})


# ---------- Run App ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
