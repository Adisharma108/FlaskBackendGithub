# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code including models folder
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Environment variable to disable debug mode (production)
ENV FLASK_ENV=production

# Run the Flask app
CMD ["python", "main.py"]
