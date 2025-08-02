# credit-card-fraud-detection

### Problem Statement

Develop a machine learning model to classify incoming credit card transactions as either fraudulent or legitimate in real-time. The system must process transaction data—such as amount, time, merchant, and location—and render a decision to approve or deny the transaction within a strict, low-latency window. The primary objective is to maximize the identification of fraudulent transactions while minimizing the number of incorrectly declined legitimate transactions to prevent financial loss and ensure a seamless customer experience.

### Data

The dataset used to train and test the models can be found on Kaggle [Fraud-Detection](https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection).
```shell
bash scripts/download_data.sh
```

Note: This is not real-data as credit card companies would not share information or datasets.

To understand the dataset, an exploratory data analysis is conducted and further relevant features for training the model are selected. This can found [here](https://github.com/soham-chitnis10/credit-card-fraud-detection/blob/main/eda.ipynb)
### Setup

Setup the envirnonment, install the prehooks and publish the docker image for MLFlow server
```shell
make setup
make publish-mlflow
```

### Training model

For training models, this requires an AWS EC2 Instance which is public and asscessible from anywhere to host the MLflow server. Additionally, this used AWS RDS with Postgres SQL for MLflow server backend store. To store the artifacts, an AWS S3 Bucket must be created. After creating the EC2 instance, AWS S3 bucket and RDS database. Remember to configure the RDS with your EC2 instance such that the EC2 instance can access, i.e. first create an EC2 instance and S3 bucket later create the RDS instance. While training this uses Prefect for workflow orchestration. Note: Prefect is not deployed on any server rather set up on the local machine.

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
The web server is build using FastAPI for model serving.
Run the below command to build and publish the docker image for the web server to the ECR repository
```bash
make publish-web-server
```

The web server is  manually deployed using an AWS EC2 instance. Install docker, configure AWS credentials and pull the published Docker image of the web server in EC2 instance. Note: It is required to set environment variables: `MLFLOW_TRACKING_SERVER` and the AWS credentials.

To start the web-server,

```bash
docker run -p 8000:8000 \
            -d --rm \
            -e MLFLOW_TRACKING_SERVER_URI=remote_server_tracking_uri \
            -e AWS_ACCESS_KEY=your_access_key \
            -e AWS_SECRET_ACCESS_KEY=asecret_key \
            -e AWS_DEFAULT_REGION=default_region
            IMAGE_NAME
```
The API of the web server would  `http://your-EC2-instance-public-DNS:8000/predict`

### Testing and Monitoring

For monitoring, Evidently AI is used. Create an account on Evidently cloud, API Key and Organization. Set them as environment varaibles `EVIDENTLY_API_KEY` and `ORG_ID`.

To get predictions and monitor the results, use the following command

`bash scripts/eval.sh TEST_SET PREDICTION_FILE WEB_SERVER_API BATCH_SIZE  NUM_WORKERS`
