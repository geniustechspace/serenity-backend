from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from chat import IntentClassifier
import logging

# Initialize the router
router = APIRouter()

intent_file = "./intent2.json"

# Initialize the classifier (global so it doesn't reload for every request)
intent_classifier = IntentClassifier()

# Attempt to load the model
try:
    intent_classifier.load_model(intent_file)
    logging.info("Model successfully loaded.")
except FileNotFoundError:
    logging.error("Model not found. Training the model.")
    intent_classifier.train_model(intent_file)
    logging.info("Model trained and saved successfully.")


# Define request model
class UserInput(BaseModel):
    text: str


# Define route for intent classification and response
@router.post("/predict/")
async def predict_intent(user_input: UserInput):
    try:
        result = intent_classifier.predict_intent(user_input.text)
        logging.info(f"User input: {user_input.text}, Predicted: {result}")
        return result
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Define route for model evaluation
@router.get("/evaluate/")
async def evaluate_model():
    try:
        evaluation_report = intent_classifier.evaluate_model()
        logging.info("Model evaluation performed.")
        return evaluation_report
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Welcome route for testing
@router.get("/")
async def root():
    return {"message": "Welcome to the Intent Classification API"}
