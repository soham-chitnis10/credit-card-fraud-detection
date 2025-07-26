""" "
Model service for credit card fraud detection.
This service handles model loading, preprocessing of input data, and prediction.
"""

import os
import json
import base64
import pickle
from datetime import datetime

import boto3
import numpy as np
import mlflow
import mlflow.artifacts
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
    model_version = get_lastest_model_version().version
    print(f"Loading model from: {model_version}")
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/CreditCardFraudDetector-MLP/{model_version}"
    )
    return model, model_version


def base64_decode(encoded_data):
    """
    Decode base64 encoded data to a dictionary.
    Args:
        encoded_data (str): Base64 encoded string.
    Returns:
        dict: Decoded transaction event data.
    """
    decoded_data = base64.b64decode(encoded_data).decode("utf-8")
    transaction_event = json.loads(decoded_data)
    return transaction_event


class ModelService:
    """
    Model service for credit card fraud detection.
    This service handles model loading, preprocessing of input data, and prediction.
    """

    def __init__(self, model, scaler, model_version=None, callbacks=None):
        """
        Initialize the ModelService with the model, scaler, and optional callbacks.
        Parameters:
            model: The trained model for prediction.
            scaler: The scaler for preprocessing input data.
            model_version: The version of the model (optional).
            callbacks: A list of callback functions (optional).
        """
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
        """
        AWS Lambda handler for processing Kinesis events.
        Args:
            event (dict): The event data from Kinesis.
        Returns:
            dict: A dictionary containing predictions.
        """
        prediction_events = []
        for record in event["Records"]:
            transaction_event = base64_decode(record["kinesis"]["data"])
            prediction = self.predict(transaction_event)
            prediction_event = {
                "prediction": prediction,
                "model_version": self.model_version,
                "trans_num": transaction_event['trans_num'],
            }
            for callback in self.callbacks:
                callback(prediction_event)
            prediction_events.append(prediction_event)
        return {"predictions": prediction_events}


class KinesisCallback:
    def __init__(self, kinesis_client, prediction_stream_name):
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event):
        trans_num = prediction_event['trans_num']

        self.kinesis_client.put_record(
            StreamName=self.prediction_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=trans_num,
        )


def create_kinesis_client():
    endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')

    if endpoint_url is None:
        return boto3.client('kinesis')

    return boto3.client('kinesis', endpoint_url=endpoint_url)


def init(prediction_stream_name: str, test_run: bool):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model, model_version = load_model()
    scaler = get_standard_scaler()
    callbacks = []

    if not test_run:
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(
        model=model, scaler=scaler, model_version=model_version, callbacks=callbacks
    )

    return model_service
