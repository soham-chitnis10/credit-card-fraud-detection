#!/bin/bash

set -a
source .env

python eval.py --data_path $1 --output_path $2 --num_workers $5 --batch_size $4 --url $3
python monitoring.py --reference_data_path data/fraudTrain.csv --current_data_path $2
