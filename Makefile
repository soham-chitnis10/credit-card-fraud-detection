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

build:
		docker build -f 'lambda.dockerfile' -t ${LOCAL_IMAGE_NAME} '.'
