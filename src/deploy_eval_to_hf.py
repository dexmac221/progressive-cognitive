import os
from huggingface_hub import HfApi

# Configuration
TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "progressive-cognitive-evaluation"
USERNAME = "dexmac"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

api = HfApi(token=TOKEN)

print(f"Creating Space {REPO_ID}...")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True
    )
    print("Space created/found successfully.")
except Exception as e:
    print(f"Error creating Space: {e}")
    exit(1)

# Create Dockerfile
dockerfile_content = """
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Fix for libgomp: Invalid value for environment variable OMP_NUM_THREADS
ENV OMP_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

# Copy evaluation scripts
COPY evaluation_framework.py .
COPY run_real_evaluation.py .

# Run a dummy HTTP server in the background to pass Hugging Face health check (port 7860)
# and run the evaluation script in the foreground
CMD python -m http.server 7860 & python run_real_evaluation.py
"""

with open("Dockerfile_eval", "w") as f:
    f.write(dockerfile_content)

# Create README.md for the Space
readme_content = f"""---
title: Progressive Cognitive Evaluation
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Progressive Cognitive Architecture - Evaluation

This Space runs the comparative evaluation between Base, Flat-LoRA, and Progressive-LoRA models.
"""

with open("SPACE_README_EVAL.md", "w") as f:
    f.write(readme_content)

print("Uploading files to Space...")
try:
    api.upload_file(
        path_or_fileobj="SPACE_README_EVAL.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="Dockerfile_eval",
        path_in_repo="Dockerfile",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="src/evaluation_framework.py",
        path_in_repo="evaluation_framework.py",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="src/run_real_evaluation.py",
        path_in_repo="run_real_evaluation.py",
        repo_id=REPO_ID,
        repo_type="space"
    )
    print("Files uploaded successfully.")
except Exception as e:
    print(f"Error uploading files: {e}")
    exit(1)

print("Requesting GPU hardware (T4 small)...")
try:
    api.request_space_hardware(repo_id=REPO_ID, hardware="t4-small")
    print("GPU hardware requested successfully. Evaluation will start shortly!")
    print(f"You can monitor logs here: https://huggingface.co/spaces/{REPO_ID}?logs=container")
except Exception as e:
    print(f"Error requesting hardware: {e}")
    print("Make sure you have a valid payment method on Hugging Face.")

# Add secrets to allow the script to upload the results and pause the space
try:
    api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=TOKEN)
    api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value=f"{USERNAME}/progressive-cognitive-results")
    print("Secrets added successfully.")
except Exception as e:
    print(f"Error adding secrets: {e}")

print("\nDone! The evaluation will run and results will be saved.")
