FROM python:3.11-slim AS base

# System deps:
# - ffmpeg: required by ffmpeg-python usage
# - ca-certificates: TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python packaging)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Create venv and install deps (no project code yet)
# --frozen uses uv.lock exactly (good for reproducibility)
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv sync --frozen --no-dev

# Now copy the rest of the repo
COPY . .

# Make the venv the default Python
ENV PATH="/app/.venv/bin:${PATH}"

# Optional: make Python output unbuffered (useful in containers)
ENV PYTHONUNBUFFERED=1

# Default command: show help (override in docker run)
CMD ["tennis-cut", "--help"]