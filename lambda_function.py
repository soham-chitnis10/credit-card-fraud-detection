import os

import model_service

PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'


model_service_instance = model_service.init(
    prediction_stream_name=PREDICTIONS_STREAM_NAME,
    test_run=TEST_RUN,
)


def lambda_handler(event, context):
    # pylint: disable=unused-argument
    return model_service_instance.lambda_handler(event)
