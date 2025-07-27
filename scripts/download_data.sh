#!/bin/bash
# Download the dataset for credit card fraud detection
curl -L -o fraud-detection.zip https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection
unzip fraud-detection.zip -d data/
rm fraud-detection.zip
