# Galileo web app: film or upload an experiment, get the measurement back.
#   docker build -t galileo .
#   docker run -p 7860:7860 -e MODEL_API_KEY=... galileo        # without the key: demo mode
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY galileo ./galileo
COPY docs ./docs
RUN pip install --no-cache-dir -e ".[web]"

# Writable places for a non-root user (Hugging Face Spaces runs as uid 1000).
ENV GALILEO_DATA=/tmp/galileo-web MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn galileo.web:app --host 0.0.0.0 --port ${PORT}"]
