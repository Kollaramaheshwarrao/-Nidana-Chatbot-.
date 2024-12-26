import json
import random
import logging
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure required NLTK data is available
nltk.download('punkt')
nltk.download('wordnet')

# Initialize app
app = Flask(__name__)

# Load intents
with open("data/intents.json", "r") as file:
    intents = json.load(file)["intents"]

# Preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return " ".join(lemmatized_tokens)

# Prepare training data
training_sentences = []
labels = []
for intent_data in intents:
    for pattern in intent_data["patterns"]:
        training_sentences.append(preprocess(pattern))
        labels.append(intent_data["tag"])

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)
model = MultinomialNB()
model.fit(X, labels)
logging.basicConfig(level=logging.INFO)

def get_response(user_input):
    user_input_preprocessed = preprocess(user_input)
    input_vectorized = vectorizer.transform([user_input_preprocessed])
    predicted_intent = model.predict(input_vectorized)[0]

    logging.info(f"User Input: {user_input}, Predicted Intent: {predicted_intent}")
    ...
# Get response based on intent
def get_response(user_input):
    user_input_preprocessed = preprocess(user_input)
    input_vectorized = vectorizer.transform([user_input_preprocessed])
    predicted_intent = model.predict(input_vectorized)[0]
   

    # Check for greetings first
    if user_input.lower() in ["hi", "hello", "thanks"]:
        return "Hi there! How are you feeling today?"

    # Find the correct response based on predicted intent
    for intent_data in intents:
        if intent_data["tag"] == predicted_intent:
            return random.choice(intent_data["responses"])
    
    return "I'm sorry, I didn't understand that. Could you try again?"

    
# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_response = get_response(user_message)
    return jsonify({"response": bot_response})

# Run server
if __name__ == "__main__":
    app.run(debug=True)
while True:
    user_input = input("You: ").strip()
    
    # Check if the input is empty
    if not user_input:
        print("Bot: Please enter a valid message.")
        continue  # Skip the rest of the loop and ask for input again
    
    # Process the input and generate a response
    response = response(user_input)  # Assuming this is your function
    print(f"Bot: {response}")
print(f"Predicted Intent: {predicted_intent}")  # Debug output to verify correct prediction

