FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Copy the basic_neuralnetwork.py script into the container
COPY app.py .
COPY scripts/basic_neuralnetwork.py .
RUN mv basic_neuralnetwork.py app.py
# Expose the port for the ASGI server
EXPOSE 8000

# Start the BottleRocket ASGI server with Uvicorn
CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000"]

