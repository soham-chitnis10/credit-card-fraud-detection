FROM python:3.12-slim
RUN pip install pipenv
COPY [ "Pipfile", "Pipfile.lock", "./" ]
RUN pipenv install --system --deploy
COPY [ "./web_server", "/web_server" ]
WORKDIR /web_server
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
