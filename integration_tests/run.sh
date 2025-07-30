#!/bin/bash

if [ -z "${MLFLOW_TRACKING_URI}" ]; then
    echo "MLFLOW_TRACKING_URI is not set. Please set it before running the script."
    exit 1
fi
if [ -z "${AWS_ACCESS_KEY_ID}" ]; then
    echo "AWS_ACCESS_KEY_ID is not set. Please set it before running the script."
    exit 1
fi
if [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "AWS_SECRET_ACCESS_KEY is not set. Please set it before running the script."
    exit 1
fi
if [ -z "${AWS_DEFAULT_REGION}" ]; then
    echo "AWS_DEFAULT_REGION is not set. Please set it before running the script."
    exit 1
fi

docker run -d --rm \
    -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
    -p 8000:8000 \
    web-service-credit-card-fraud-detection:latest

sleep 10

python integration_tests/test_model_prediction.py

if [ $? -ne 0 ]; then
    echo "Integration tests failed."
    docker stop $(docker ps -q --filter ancestor=web-service-credit-card-fraud-detection:latest)
    exit 1
fi

docker stop $(docker ps -q --filter ancestor=web-service-credit-card-fraud-detection:latest)
echo "Docker container stopped."
echo "Integration tests completed."
