FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NAIVE_BAYES_MODEL_PATH=/app/models/phishing_model_artifact.pkl
ENV NAIVE_BAYES_VECTORIZER_PATH=/app/models/phishing_vectorizer.pkl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY data_science/SMSModel/artifacts/phishing_model_artifact.pkl models/phishing_model_artifact.pkl
COPY data_science/SMSModel/artifacts/phishing_vectorizer.pkl models/phishing_vectorizer.pkl

RUN useradd --create-home --shell /usr/sbin/nologin safefam \
    && chown -R safefam:safefam /app

USER safefam

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
