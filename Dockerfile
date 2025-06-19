FROM python:3.9

WORKDIR /code

# Create cache directories with proper permissions
RUN mkdir -p /tmp/huggingface_cache /tmp/huggingface_home /tmp/huggingface_datasets && \
    chmod 777 /tmp/huggingface_cache /tmp/huggingface_home /tmp/huggingface_datasets

# Set environment variables for Hugging Face cache
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV HF_HOME=/tmp/huggingface_home
ENV HF_DATASETS_CACHE=/tmp/huggingface_datasets

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app.py /code/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]