# Start from a Python 3.10 slim image
# slim = smaller size, no unnecessary system packages
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# If requirements don't change, Docker skips reinstalling
COPY ./req.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build
# So it's baked into the image, not downloaded at runtime
RUN python -m spacy download en_core_web_sm

# Copy the rest of your code
COPY . /app

# Create directory for graph cache
RUN mkdir -p graph_cache

# Expose the FastAPI port
EXPOSE 7860

# Start both FastAPI and Gradio using a startup script
CMD ["python", "start.py"]