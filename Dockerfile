FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from kiwipiepy import Kiwi; assert Kiwi().tokenize('installation check')"

COPY app app
COPY data_science/SMSModel/modeling data_science/SMSModel/modeling
COPY data_science/SMSModel/tokenization data_science/SMSModel/tokenization
COPY data_science/SMSModel/artifacts/stacking data_science/SMSModel/artifacts/stacking

RUN useradd --create-home --shell /usr/sbin/nologin safefam \
    && chown -R safefam:safefam /app \
    && chown -R root:root /app/data_science/SMSModel/artifacts/stacking \
    && chmod -R a-w /app/data_science/SMSModel/artifacts/stacking

USER safefam

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
