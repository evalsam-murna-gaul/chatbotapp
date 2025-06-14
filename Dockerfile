# Use an official PyTorch base image with Python 3.10
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy your requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "127.0.0.0", "--port", "8000"]


