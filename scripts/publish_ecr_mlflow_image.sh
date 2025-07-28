#!/bin/bash

export ECR_REPOSITORY="credit-card-fraud-detection"

aws ecr describe-repositories --repository-names $ECR_REPOSITORY

if [ $? -ne 0 ]; then
    echo "Repository $ECR_REPOSITORY does not exist. Creating it now..."
    aws ecr create-repository --repository-name $ECR_REPOSITORY
    if [ $? -ne 0 ]; then
        echo "Failed to create repository $ECR_REPOSITORY."
        exit 1
    fi
else
    echo "Repository $ECR_REPOSITORY already exists."
fi


REPO_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY --query "repositories[0].repositoryUri" --output text)
echo "Repository URI: $REPO_URI"
docker tag mlflow:latest $REPO_URI:mlflow
docker push --platform linux/x86_64 $REPO_URI:mlflow
