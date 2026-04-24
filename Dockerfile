# Dockerfile for HuggingFace Spaces / general cloud deployment of the
# Portfolio Backtest Engine Streamlit dashboard.
#
# Build: docker build -t portfolio-backtest-dashboard .
# Run:   docker run -p 7860:7860 portfolio-backtest-dashboard
#
# HuggingFace Spaces note: SDK must be "docker" in README metadata.
# Default app port is 7860 on HF Spaces.

FROM python:3.12-slim

WORKDIR /app

# System deps for matplotlib, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (Docker layer caching)
COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy application
COPY . .

# HF Spaces convention: expose 7860
EXPOSE 7860

# Run Streamlit on the HF-expected port, binding to 0.0.0.0
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
