FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Erst nur die Abhängigkeiten installieren, damit der Layer bei Codeänderungen gecacht bleibt
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install ".[dev]" \
    && pip uninstall -y adventure-planner

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
