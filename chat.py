import json
import logging
import random
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

logging.basicConfig(
    filename="chat.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class IntentClassifier:
    def __init__(self):
        self.model = None
        self.intents = None
        self.X_test = None
        self.y_test = None

    def load_data(self, intent_file):
        """Load the intents from the given file."""
        with open(intent_file, "r") as f:
            self.intents = json.load(f)

        data = []
        labels = []

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                data.append(pattern)
                labels.append(intent["tag"])

        return pd.DataFrame({"text": data, "intent": labels})

    def train_model(self, intent_file, test_size=0.2):
        """Train the model and save it to disk."""
        # Load the intents
        data = self.load_data(intent_file)

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            data["text"], data["intent"], test_size=test_size, random_state=42
        )

        # Create a pipeline with a TF-IDF Vectorizer and an SVM classifier
        self.model = make_pipeline(TfidfVectorizer(), SVC(kernel="linear"))
        self.model.fit(X_train, y_train)

        # Save the test set for evaluation
        self.X_test = X_test
        self.y_test = y_test

        # Save the trained model
        joblib.dump(self.model, "intent_model.pkl")
        logging.info("Model trained and saved successfully.")

    def load_model(self, intent_file):
        """Load the model from disk and ensure the intents are also loaded."""
        if os.path.exists("intent_model.pkl"):
            # Load the model from disk
            self.model = joblib.load("intent_model.pkl")
            logging.info("Model loaded from disk.")

            # Ensure the intents are loaded
            self.load_data(intent_file)
            logging.info("Intents data loaded.")
        else:
            raise FileNotFoundError(
                "Model file not found. Please train the model first."
            )

    def predict_intent(self, text):
        """Predict the intent and return the appropriate response."""
        if not self.model:
            raise ValueError("Model is not loaded. Please train or load the model.")

        # Perform prediction
        predicted_intent = self.model.predict([text])[0]

        # Get the response for the predicted intent
        response = self.get_response(predicted_intent)
        return {"intent": predicted_intent, "response": response}

    def get_response(self, intent):
        """Retrieve a random response for the given intent."""
        if not self.intents:
            logging.error("Intents data not loaded.")
            return "Sorry, I don't understand."

        for item in self.intents["intents"]:
            if item["tag"] == intent:
                return random.choice(item["responses"])

        logging.error(f"No response found for intent: {intent}")
        return "Sorry, I don't understand."

    def evaluate_model(self):
        """Evaluate the model on the test set and return the performance report."""
        if not self.X_test or not self.model:
            raise ValueError("Model or test data is missing. Please train the model.")

        predictions = self.model.predict(self.X_test)
        report = classification_report(self.y_test, predictions, output_dict=True)
        return report


if __name__ == "__main__":
    intent_classifier = IntentClassifier()

    # Train model if needed, otherwise load the existing model
    intent_file = "./intent2.json"

    if not os.path.exists("intent_model.pkl"):
        intent_classifier.train_model(intent_file, test_size=0.2)
    else:
        intent_classifier.load_model(intent_file)

    # Evaluate the model
    results = intent_classifier.evaluate_model()
    print(results)

    # Simulate user inputs
    user_inputs = [
        "Hi",
        "Who are you?",
        "What can you do?",
        "I'm feeling sad today",
        "What is mental health?",
    ]

    for i in user_inputs:
        intent = intent_classifier.predict_intent(i)
        print(intent)
