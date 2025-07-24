FROM python:3.12-slim

RUN pip3 install mlflow boto3 psycopg2-binary

EXPOSE 5000


CMD mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $BACKEND_STORE_URI --default-artifact-root $S3_ARTIFACT_ROOT --serve-artifacts --artifacts-destination $S3_ARTIFACT_ROOT
