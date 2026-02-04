FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
