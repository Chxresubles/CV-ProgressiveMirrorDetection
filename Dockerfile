FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml .
COPY ./cvprogressivemirrordetection ./cvprogressivemirrordetection

RUN pip install --no-cache-dir --no-compile . --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY ./output/model.pkl ./output/
COPY ./scripts/score.py .
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

CMD ["python", "score.py"]
