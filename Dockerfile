FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Install the package in editable mode (makes models.py importable)
RUN pip install --no-cache-dir -e .

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Default: run the API server (no web interface)
ENV ENABLE_WEB_INTERFACE=false

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
