# Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /rag-cache

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "ui.py"]