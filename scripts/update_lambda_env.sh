#!/bin/bash

set -a
source ../.env
export LAMBDA_FUNCTION="lambda-function-credit-card-fraud-detection"
export PREDICTIONS_STREAM_NAME="stg-trans-event-prediction-credit-card-fraud-detection"
variables="{PREDICTIONS_STREAM_NAME=${PREDICTIONS_STREAM_NAME}, MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}}"

aws lambda update-function-configuration --function-name ${LAMBDA_FUNCTION} --environment "Variables=${variables}"
