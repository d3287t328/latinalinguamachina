[200~# Use the official Python image as the base image
FROM python:latest

# Set the working directory
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install micrograd

# Copy the basic_neuralnetwork.py script into the container
COPY basic_neuralnetwork.py .

# Expose the port for the ASGI server
EXPOSE 8000

# Start the BottleRocket ASGI server with Uvicorn
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "bottlerocket:app"]

