
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
