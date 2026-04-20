# ── Stage 1: resolve the Git LFS model file ──────────────────────────────────
# We need a real Git repository so that `git lfs pull` can contact the LFS
# server and download the actual binary.  A plain `COPY . .` only copies the
# LFS pointer text, so we clone the repo here instead.
FROM python:3.10-slim AS lfs-resolver

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository normally so files are checked out, then pull LFS objects
# to resolve the pointer at backend/best_model.pth into the real binary.
RUN git clone https://github.com/Helouniverse/TB-detection-web-application.git /repo \
    && cd /repo \
    && git lfs install \
    && git lfs pull --include="backend/best_model.pth"

# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install required system packages, e.g., libgl1 for OpenCV if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install python dependencies, pulling PyTorch from CPU-only index
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the application source code (LFS pointer files come from the build context)
COPY . .

# Overwrite the LFS pointer with the real model binary resolved in stage 1
COPY --from=lfs-resolver /repo/backend/best_model.pth ./backend/best_model.pth

# Railway provides PORT env var, but 8000 is default local fallback
ENV PORT=8000
EXPOSE $PORT

CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT
