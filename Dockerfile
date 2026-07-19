FROM python:3.11-slim

WORKDIR /app

# System deps: libsndfile for soundfile, ffmpeg for librosa's audio backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller/faster than the default CUDA
# build, and Railway has no GPU anyway) — then the rest of requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the app AND the bundled adapter folder (commit ./adapter into git —
# see README notes below)
COPY . .

# Ensures print()/tracebacks flush immediately to Railway's log stream
# instead of being buffered and possibly lost on a crash.
ENV PYTHONUNBUFFERED=1

# Railway injects $PORT at runtime; app.py reads it directly
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
