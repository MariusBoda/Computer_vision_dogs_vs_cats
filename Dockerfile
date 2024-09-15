# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create a working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary Python packages
# If you have additional dependencies, you can add them to a requirements.txt file
RUN pip install --upgrade pip && \
    pip install torch torchvision

# Specify the device as GPU
ENV DEVICE=cpu

# Run the Python script when the container launches
CMD ["python", "model.py"]
