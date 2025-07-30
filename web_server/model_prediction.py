""" "
Model service for credit card fraud detection.
This service handles model loading, preprocessing of input data, and prediction.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import mlflow
import mlflow.artifacts

FEATURES = [
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "trans_time",
    "month",
    "category_food_dining",
    "category_gas_transport",
    "category_grocery_net",
    "category_grocery_pos",
    "category_health_fitness",
    "category_home",
    "category_kids_pets",
    "category_misc_net",
    "category_misc_pos",
    "category_personal_care",
    "category_shopping_net",
    "category_shopping_pos",
    "category_travel",
]


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
    """
    Get the standard scaler from the latest model version.
    Returns:
        StandardScaler: The standard scaler used for preprocessing.
    """
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
    """
    Load the latest model version from MLflow.
    Returns:
        tuple: The loaded model and its version.
    """
    model_info = get_lastest_model_version()
    print("Loading latest model version")
    model = mlflow.pyfunc.load_model(model_uri=model_info.source)
    return model, model_info.version


class ModelService:
    """
    Model service for credit card fraud detection.
    This service handles model loading, preprocessing of input data, and prediction.
    """

    def __init__(self, model, scaler, model_version=None):
        """
        Initialize the ModelService with the model, scaler, and optional callbacks.
        Parameters:
            model: The trained model for prediction.
            scaler: The scaler for preprocessing input data.
            model_version: The version of the model (optional).
        """
        self.model = model
        self.scaler = scaler
        self.model_version = model_version

    @classmethod
    def preprocess(cls, transaction_event: dict):
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

        transaction_event['trans_time'] = dt.hour * 3600 + dt.minute * 60 + dt.second
        transaction_event["month"] = dt.month
        features = []
        for feature in FEATURES:
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

    def predict(self, transaction_event: dict):
        """
        Predict if the transaction is fraudulent or not.
        """
        features = self.preprocess(transaction_event)
        features = self.scaler.transform(features)
        prediction = self.model.predict(features).argmax(axis=1).item()

        return prediction  # Return the prediction result


def init():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model, model_version = load_model()
    print(f"Loaded model version: {model_version}")
    scaler = get_standard_scaler()
    print("Loaded scaler")
    model_service = ModelService(
        model=model, scaler=scaler, model_version=model_version
    )

    return model_service
