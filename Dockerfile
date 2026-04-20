FROM python:3.10-slim

WORKDIR /app

# Install required system packages, e.g., libgl1 for OpenCV if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install python dependencies, pulling PyTorch from CPU-only index
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the application source code (including LFS pointer files)
COPY . .

# Resolve Git LFS pointers — pulls the actual model binary from LFS storage
RUN git lfs install && git lfs pull

# Railway provides PORT env var, but 8000 is default local fallback
ENV PORT=8000
EXPOSE $PORT

CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT
