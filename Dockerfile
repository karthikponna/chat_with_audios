# Use the official uv image (contains uv binary + a Debian base)
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.8.22-debian-slim

# metadata
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Avoid uv installer modifying user PATH during build
ENV UV_NO_MODIFY_PATH=1

WORKDIR /app

# Install runtime/system deps (ffmpeg, build tools for wheels, git if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      git \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (cache-friendly)
COPY pyproject.toml pyproject.toml
# If you have a uv.lock, copy it (recommended)
COPY uv.lock uv.lock

# Install transitive dependencies (but NOT the project) to leverage layer caching.
# If you do not have uv.lock, omit --locked so uv may resolve and create a lock.
RUN uv sync --locked --no-install-project || uv sync --no-install-project

# Now copy the project (source code)
COPY . /app

# Install the project itself into the environment (this step installs your package / local deps)
# Using --locked is recommended in CI/production; remove --locked if the lockfile may not match.
RUN uv sync --locked || uv sync

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit using uv run so Streamlit is executed inside uv's environment
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
