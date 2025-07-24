import base64
import json
import os
import pickle
from datetime import datetime

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import utils

load_dotenv()


def get_lastest_model_version():
    """
    Get the latest model version from the registered model.

    Returns:
        str: The latest model version.
    """
    registered_models = mlflow.search_registered_models(
        filter_string="name='CreditCardFraudDetector-MLP'"
    )[0].latest_versions
    if registered_models is None or len(registered_models) == 0:
        raise ValueError(
            "No registered model found with the name 'CreditCardFraudDetector-MLP'."
        )

    return registered_models[0]


def get_standard_scaler():
    run_id = get_lastest_model_version().run_id
    print(f"Loading scaler from run_id: {run_id}")
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler.pkl", dst_path="artifacts"
    )
    print(path)

    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def load_model():
    model_version = get_lastest_model_version().version
    print(f"Loading model from: {model_version}")
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/CreditCardFraudDetector-MLP/{model_version}"
    )
    return model, model_version


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode("utf-8")
    transaction_event = json.loads(decoded_data)
    return transaction_event


class ModelService:
    def __init__(self, model, scaler, model_version=None, callbacks=None):
        self.model = model
        self.scaler = scaler
        self.model_version = model_version
        self.callbacks = callbacks or []

    @classmethod
    def preprocess(cls, transaction_event):
        """
        Preprocess the transaction event data.

        Parameters:
        transaction_event (dict): The transaction event data.

        Returns:
        np.ndarray: The preprocessed features.
        """
        dt = datetime.strptime(
            transaction_event["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S"
        )
        transaction_event["trans_time"] = dt.hour * 3600 + dt.minute * 60 + dt.second
        transaction_event["month"] = dt.month
        features = []
        for feature in utils.features:
            if feature.startswith("category"):
                cat_type = feature.split("category_")[1]
                if transaction_event["category"] == cat_type:
                    features.append(1)
                else:
                    features.append(0)
            else:
                features.append(transaction_event[feature])
        features = np.array(features, dtype=np.float32)
        features = features.reshape(1, -1)
        return features

    def predict(self, transaction_event):
        """
        Predict if the transaction is fraudulent or not.
        """
        features = self.preprocess(transaction_event)
        features = features.reshape(1, -1)  # Reshape for single sample prediction
        features = self.scaler.transform(features)
        prediction = self.model.predict(features).argmax(axis=1).item()

        return prediction  # Return the prediction result

    def lambda_handler(self, event):

        prediction_events = []
        for record in event["Records"]:
            transaction_event = base64_decode(record["kinesis"]["data"])
            prediction = self.predict(transaction_event)
            prediction_events.append(
                {"prediction": prediction, "model_version": self.model_version}
            )
