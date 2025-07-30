LOCAL_IMAGE_NAME:=web-service-credit-card-fraud-detection
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

build-web-service:
		docker build -f 'web-service.dockerfile' -t ${LOCAL_IMAGE_NAME} .

integration-test: build-web-service
		bash integration_tests/run.sh

publish-mlflow: build-mlflow
		bash scripts/publish_ecr_mlflow_image.sh

publish-web-service: build-web-service
		bash scripts/publish_ecr_web_service_image.sh
