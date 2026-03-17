FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System tools
RUN apt-get update && apt-get install -y git wget nano && rm -rf /var/lib/apt/lists/*

# Set Python path so it can see the /app directory for imports
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 1. Copy requirements (Path is relative to the Root where you run the command)
COPY Docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2. Copy the src folder (contains datasets.py, models.py, etc.)
COPY src/ /app/src/

# 3. Copy the project folder contents (train_phase1.py, eval_phase1.py)
COPY project/ /app/

# 4. Copy config.yaml from project folder to the root of /app
# (Your project has config.yaml inside 'project', so we put it in /app)
COPY project/config2.yaml /app/config2.yaml

# 5. Create output directories
RUN mkdir -p /outputs/logs /outputs/checkpoints

# Start training (No /configs/ needed if config is now in /app)
CMD ["python", "train_phase2_fl.py", "--config", "config2.yaml"]