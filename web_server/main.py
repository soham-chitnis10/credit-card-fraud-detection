import model_prediction
import transaction_event_model
from fastapi import FastAPI

app = FastAPI()

model_service_instance = model_prediction.init()


@app.post("/predict")
async def predict_transaction(
    transaction_event: transaction_event_model.TransactionEvent,
):
    event = transaction_event.__dict__
    prediction = model_service_instance.predict(event)
    return {
        "trans_num": transaction_event.trans_num,
        "prediction": prediction,
        "model_version": model_service_instance.model_version,
    }
