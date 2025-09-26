FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-cache

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application using uv run (maintains consistency with your local setup)
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]