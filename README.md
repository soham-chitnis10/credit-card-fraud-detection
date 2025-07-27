# credit-card-fraud-detection

### Problem Statement

Predict in real-time a credit card transaction is fradulent or not.

### Data

The dataset used to train and test the models can be found on Kaggle [Fraud-Detection](https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection).
```shell
bash scripts/download_data.sh
```

### Setup

Setup the envirnonment, install the prehooks and publish the docker image for MLFlow server
```shell
make setup
make publish-mlflow
```

### Training model

For training models, this requires an AWS EC2 Instance which is public and asscessible from anywhere to host the MLflow server. Additionally, this used AWS RDS with Postgres SQL for MLflow server backend store. To store the artifacts, an AWS S3 Bucket must be created. After creating the EC2 instance, AWS S3 bucket and RDS database. Remember to configure the RDS with your EC2 instance such that the EC2 instance can access, i.e. first create an EC2 instance and S3 bucket later create the RDS instance.

Install Docker in EC2 Instance, Configure AWS, Log into ECR repository to access the published the docker image for running the MLFLOW server. Pull the image `mlflow` from the repository  `credit-card-fraud-detection`.

```shell
docker run -p 5000:5000 \
            -d --rm \
            -e BACKEND_STORE_URI=your_backend_store_uri \
            -e S3_ARTIFACT_ROOT=your_s3_bucket_uri \
            IMAGE_NAME
```

The above command will run the container with background mode on the server.

Now, we can train the model on our system.
```shell
export MLFLOW_TRACKING_SERVER=http://your-EC2-instance-public-DNS:5000
bash training_script.sh
```
This create a new experiment and perform grid search to find the best hyperparamters and model architecture. This script is starts a prefect server on the local system for the workflow orchestration. This is store experiment information to the backend store, save the artifacts including the model to the S3 bucket.

### Deployment
