# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    bash \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Install texteller and torch (CPU version)
RUN uv pip install --system texteller && \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Set work directory
WORKDIR /app

# Copy run script
COPY run.sh /app/run.sh

# Make run.sh executable
RUN chmod +x /app/run.sh

# Expose port
EXPOSE 8501

# Run the application
CMD ["/app/run.sh"]
