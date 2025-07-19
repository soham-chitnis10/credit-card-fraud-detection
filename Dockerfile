FROM python:3.12-slim
COPY ["Pipfile","Pipfile.lock", "./"]
RUN  pip install -U pipenv
RUN pipenv install --system --deploy
CMD ["python", "--version"]
