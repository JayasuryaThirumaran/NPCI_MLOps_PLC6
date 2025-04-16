# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Copy all files from the dataset directory on the host to the dataset directory in the container
ADD dataset/* dataset/

# Copy all trained model files (with .pkl extension) from the host to the trained_model directory in the container
ADD trained_model/*.pkl trained_model/

# Copy all requirement text files from the requirements directory on the host to the requirements directory in the container
ADD requirements/*.txt requirements/

# Install the Python packages specified in the api_requirements.txt file
RUN pip install -r requirements/api_requirements.txt

# Copy all Python files from the current directory on the host to the current directory in the container
ADD *.py ./

# Expose port 8080 to allow external access to the application running in the container
EXPOSE 8080

# Set the command to run the application when the container starts
CMD ["python", "app.py"]
