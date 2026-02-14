import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data=pd.read_csv("bot.csv")
x=data['q']
y=data['a']

# Convert text to vectors
vectorizer =CountVectorizer()
X_vec = vectorizer.fit_transform(x)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_vec, y)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)
    return prediction[0]

while True:
    query = input("You: ")
    if query.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(query)
    print("Chatbot:", response)




