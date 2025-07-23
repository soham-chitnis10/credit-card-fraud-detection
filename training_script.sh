#! /bin/bash
# This script is used to run the training process for the credit card fraud detection model.
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
prefect server start &
sleep 10

python train.py --epochs 20 --grid_search --register_model --use_cpu
# The script runs the training with specified parameters, enabling grid search and model registration

pkill -f "prefect server start"
