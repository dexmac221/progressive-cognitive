import os
from huggingface_hub import HfApi

# Configuration
TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "progressive-cognitive-baseline"
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

COPY baseline_training.py .

# Run a dummy HTTP server in the background to pass Hugging Face health check (port 7860)
# and run the training script in the foreground
CMD python -m http.server 7860 & python baseline_training.py
"""

with open("Dockerfile_baseline", "w") as f:
    f.write(dockerfile_content)

# Create README.md for the Space
readme_content = f"""---
title: Progressive Cognitive Baseline Training
emoji: 📊
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
---

# Progressive Cognitive Architecture - Baseline Training

This Space runs the training of the baseline control model (Flat-LoRA) for A/B testing.
"""

with open("SPACE_README_BASELINE.md", "w") as f:
    f.write(readme_content)

print("Uploading files to Space...")
try:
    api.upload_file(
        path_or_fileobj="SPACE_README_BASELINE.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="Dockerfile_baseline",
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
        path_or_fileobj="src/baseline_training.py",
        path_in_repo="baseline_training.py",
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
    print("GPU hardware requested successfully. Training will start shortly!")
    print(f"You can monitor logs here: https://huggingface.co/spaces/{REPO_ID}?logs=container")
except Exception as e:
    print(f"Error requesting hardware: {e}")
    print("Make sure you have a valid payment method on Hugging Face.")

# Add secrets to allow the script to upload the model and pause the space
try:
    api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=TOKEN)
    # SPACE_ID is a reserved environment variable automatically set by Hugging Face
    api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value=f"{USERNAME}/progressive-cognitive-baseline-lora")
    print("Secrets added successfully.")
except Exception as e:
    print(f"Error adding secrets: {e}")

print("\nDone! The model will be trained and saved in the repository:")
print(f"https://huggingface.co/{USERNAME}/progressive-cognitive-baseline-lora")
