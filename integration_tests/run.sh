#!/bin/bash
cd integration_tests
export LOCAL_IMAGE_NAME="stream-credit-card-fraud-detection"
export PREDICTIONS_STREAM_NAME="credit-card-fraud-detection"
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


docker compose up -d

sleep 5

aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name credit-card-fraud-detection \
    --shard-count 1
pipenv run python test_docker.py


ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

pipenv run python test_kinesis.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi


docker compose down
