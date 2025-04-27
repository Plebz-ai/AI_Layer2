# Use an official Python runtime as a parent image
FROM python:3.13-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir flask werkzeug SpeechRecognition requests python-dotenv redis azure-ai-inference azure-core grpcio pydub

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=transcription_service.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]