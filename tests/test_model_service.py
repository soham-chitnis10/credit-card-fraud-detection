from pathlib import Path

import numpy as np

import model_service


def read_text(file):
    test_directory = Path(__file__).parent

    with open(test_directory / file, "rt", encoding="utf-8") as f_in:
        return f_in.read().strip()


def test_load_model():
    model, model_version = model_service.load_model()
    assert model is not None
    assert model_version is not None


def test_get_standard_scaler():
    scaler = model_service.get_standard_scaler()
    assert scaler is not None
    assert hasattr(scaler, "mean_") and hasattr(scaler, "scale_")


def test_base64_decode():
    expected_result = {
        "Unnamed: 0": 0,
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": 2703186189652095,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": 4.97,
        "first": "Jennifer",
        "last": "Banks",
        "gender": "F",
        "street": "561 Perry Cove",
        "city": "Moravian Falls",
        "state": "NC",
        "zip": 28654,
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 3495,
        "job": "Psychologist, counselling",
        "dob": "1988-03-09",
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time": 1325376018,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
    }
    encoded_data = read_text("data.b64")
    actual_result = model_service.base64_decode(encoded_data)
    assert actual_result == expected_result


def test_preprocess():
    transaction_event = {
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": 2703186189652095,
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": 4.97,
        "first": "Jennifer",
        "last": "Banks",
        "gender": "F",
        "street": "561 Perry Cove",
        "city": "Moravian Falls",
        "state": "NC",
        "zip": 28654,
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 3495,
        "job": "Psychologist, counselling",
        "dob": "1988-03-09",
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time": 1325376018,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
    }
    expected_features = np.array(
        [
            4.97,
            36.0788,
            -81.1781,
            36.011293,
            -82.048315,
            18.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    features = model_service.ModelService.preprocess(transaction_event)
    assert (features == expected_features.reshape(1, -1)).all()


def test_model_predict():
    model, model_version = model_service.load_model()
    scaler = model_service.get_standard_scaler()
    model_service_instance = model_service.ModelService(
        model=model, scaler=scaler, model_version=model_version
    )
    encoded_data = read_text("data.b64")
    transaction_event = model_service.base64_decode(encoded_data)
    prediction = model_service_instance.predict(transaction_event)
    assert prediction == 0


def test_lambda_handler():
    model, model_version = model_service.load_model()
    scaler = model_service.get_standard_scaler()
    model_service_instance = model_service.ModelService(
        model=model, scaler=scaler, model_version=model_version
    )

    encoded_data = read_text("data.b64")

    event = {"Records": [{"kinesis": {"data": encoded_data}}]}

    actual_predictions = model_service_instance.lambda_handler(event)
    expected_prediction = {
        "predictions": [
            {
                "prediction": 0,
                "model_version": model_version,
                "trans_num": "0b242abb623afc578575680df30655b9",
            }
        ]
    }
    assert (
        actual_predictions == expected_prediction
    ), f"Expected: {expected_prediction}, but got: {actual_predictions}"
