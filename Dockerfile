FROM python:3.12-slim

WORKDIR /app

# Install system deps for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install package with dev deps (no compression — too heavy for test image)
RUN pip install --no-cache-dir -e ".[dev]"

# Copy tests
COPY tests/ ./tests/

# Default: run all tests
CMD ["pytest", "tests/", "-v", "--tb=short", "-x"]
