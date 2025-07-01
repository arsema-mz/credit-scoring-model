FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY ./src ./src
COPY ./models ./models

# Expose port 8000
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
