import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# ---------- Load Dataset Safely ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "bot.csv")

data = pd.read_csv(csv_path)

x = data['q']
y = data['a']

# ---------- Vectorize Text ----------
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(x)

# ---------- Train Model ----------
model = MultinomialNB()
model.fit(X_vec, y)

# ---------- Chatbot Function ----------
def chatbot_response(user_input):
    if not user_input.strip():
        return "Please type something."

    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)

    return prediction[0]


# ---------- Chat Loop ----------
print("Chatbot ready! Type 'exit' to quit.")

while True:
    query = input("You: ")

    if query.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break

    response = chatbot_response(query)
    print("Chatbot:", response)
