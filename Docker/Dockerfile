FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${PYTHONPATH}:/app"

COPY Docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY project/ /app/project/
COPY project/ /app/configs/

RUN mkdir -p /app/outputs_ckks/logs /app/outputs_ckks/checkpoints /app/outputs_ckks/metrics /app/outputs_ckks/stats

CMD ["python", "project/train_ckks_fl.py", "--config", "project/config_ckks.yaml"]