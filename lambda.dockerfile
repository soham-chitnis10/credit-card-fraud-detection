FROM public.ecr.aws/lambda/python:3.12
RUN pip install pipenv
COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy
COPY ["model_service.py", "lambda_function.py", "utils.py", "model.py", "./"]
CMD ["lambda_function.lambda_handler"]
