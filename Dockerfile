# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy requirements first (for cache)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Install PyTorch (CPU)
# -----------------------------
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# -----------------------------
# Install torch-geometric dependencies
# -----------------------------
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# -----------------------------
# Install remaining Python deps
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy app code & assets
# -----------------------------
COPY . .

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Streamlit config (no CORS issues)
# -----------------------------
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501

# -----------------------------
# Run app
# -----------------------------
CMD ["streamlit", "run", "app.py"]
