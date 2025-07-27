LOCAL_IMAGE_NAME:=stream-credit-card-fraud-detection
test:
		pytest tests/

setup:
		pipenv install --dev
		pre-commit install

quality_checks:
		isort .
		black .
		pylint --recursive=y .

build-mlflow:
		docker build -f 'mlflow.dockerfile' -t 'mlflow' '.'

build-lambda:
		quality_checks test
		docker build -f 'lambda.dockerfile' -t ${LOCAL_IMAGE_NAME} '.'

publish-mlflow:
		build-mlflow
		bash scripts/publish_ecr_mlflow_images.sh
