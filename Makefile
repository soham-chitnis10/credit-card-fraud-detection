
test:
		pytest tests/

setup:
		pipenv install --dev
		pre-commit install

quality_checks:
		isort .
		black .
		pylint --recursive=y .
