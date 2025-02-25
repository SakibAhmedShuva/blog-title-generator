FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .env .
COPY templates/ templates/

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
