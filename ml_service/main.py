import joblib

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

MODEL_PATH = "sms_fraud_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ============================================================
# CREATE API
# ============================================================

app = FastAPI(
    title="Swahili SMS Scam Detection API",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class SMSRequest(BaseModel):
    message: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "loaded"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_sms(request: SMSRequest):

    message = request.message

    # Convert the message into TF-IDF features
    message_tfidf = vectorizer.transform([message])

    # Predict class
    prediction = model.predict(message_tfidf)[0]

    # Get probability of scam
    scam_probability = model.predict_proba(
        message_tfidf
    )[0][1]

    # Convert numerical prediction to label
    if prediction == 1:
        classification = "scam"
    else:
        classification = "trust"

    return {
        "classification": classification,
        "scamProbability": float(scam_probability)
    }