LOCAL_IMAGE_NAME:=stream-credit-card-fraud-detection
test:
		pytest tests/

install:
		pip install -U pipenv
		pipenv install --dev

setup:	install
		pre-commit install

quality_checks:
		isort .
		black .
		pylint --recursive=y .

build-mlflow:
		docker build -f 'mlflow.dockerfile' -t 'mlflow' '.'

build-lambda:
		docker build -f 'lambda.dockerfile' -t ${LOCAL_IMAGE_NAME} .

integration-test: build-lambda
		bash integration_tests/run.sh

publish-mlflow: build-mlflow
		bash scripts/publish_ecr_mlflow_image.sh

publish-lambda: integration-test
		bash scripts/publish_ecr_lambda_image.sh
