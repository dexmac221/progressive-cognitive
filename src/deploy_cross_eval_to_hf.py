"""
Deploy Cross-Architecture Evaluation to HF Spaces.
Tests 6 models (Qwen×3 + Phi-2×3) on the same test suite.
"""

import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "progressive-cognitive-cross-eval"
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

dockerfile_content = """
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV OMP_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

COPY evaluation_framework.py .
COPY run_cross_arch_evaluation.py .

CMD python -m http.server 7860 & python run_cross_arch_evaluation.py
"""

with open("Dockerfile_cross_eval", "w") as f:
    f.write(dockerfile_content)

readme_content = """---
title: Cross-Architecture Evaluation
emoji: 🔬
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

# Cross-Architecture Evaluation — Qwen vs Phi-2

Compares 6 models on the same English test suite:
- Qwen 2.5 1.5B: Base, Flat-LoRA, Dream-LoRA
- Phi-2 2.7B: Base, Flat-LoRA, Dream-LoRA
"""

with open("SPACE_README_CROSS_EVAL.md", "w") as f:
    f.write(readme_content)

print("Uploading files to Space...")
try:
    api.upload_file(
        path_or_fileobj="SPACE_README_CROSS_EVAL.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="Dockerfile_cross_eval",
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
        path_or_fileobj="src/run_cross_arch_evaluation.py",
        path_in_repo="run_cross_arch_evaluation.py",
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
    print("GPU hardware requested successfully!")
    print(f"Monitor logs: https://huggingface.co/spaces/{REPO_ID}?logs=container")
except Exception as e:
    print(f"Error requesting hardware: {e}")

try:
    api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=TOKEN)
    api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value=f"{USERNAME}/progressive-cognitive-results")
    print("Secrets added successfully.")
except Exception as e:
    print(f"Error adding secrets: {e}")

print(f"\nDone! Evaluation will run on T4 GPU.")
print(f"Logs: https://huggingface.co/spaces/{REPO_ID}?logs=container")
print(f"Results will be saved to: {USERNAME}/progressive-cognitive-results")
