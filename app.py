from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load and train model
data = pd.read_csv(r"C:\Users\medap\OneDrive - Aditya Educational Institutions\Documents\bot.csv")
x = data['q']
y = data['a']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(x)

model = MultinomialNB()
model.fit(X_vec, y)

# Chatbot function
def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)
    return prediction[0]

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json["msg"]
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
