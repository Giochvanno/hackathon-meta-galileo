# Hugging Face Space: builds Isochron AI straight from the public GitHub repository.
# "Factory rebuild" in the Space settings picks up the latest commit.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Giochvanno/hackathon-meta-galileo /app
WORKDIR /app
RUN pip install --no-cache-dir -e ".[web]"

ENV GALILEO_DATA=/tmp/galileo-web MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn galileo.web:app --host 0.0.0.0 --port ${PORT}"]
