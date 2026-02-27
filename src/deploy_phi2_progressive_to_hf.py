"""
Deploy Progressive Cognitive (Dream Pruning) for Phi-2 2.7B to HF Spaces.
"""

import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "progressive-cognitive-phi2-dream"
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

COPY progressive_llm_cognitive_phi2.py .

CMD python -m http.server 7860 & python progressive_llm_cognitive_phi2.py
"""

with open("Dockerfile_phi2_dream", "w") as f:
    f.write(dockerfile_content)

readme_content = f"""---
title: Progressive Cognitive Phi2 Dream
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Progressive Cognitive Architecture — Phi-2 2.7B + Dream Pruning

Cross-architecture validation: same method, different model family.

| | Qwen 2.5 1.5B | Phi-2 2.7B |
|---|---|---|
| Company | Alibaba | Microsoft |
| Params | 1.5B | 2.7B |
| License | Apache 2.0 | MIT |
| Method | 4-phase + SVD Dream Pruning | **Identical** |
"""

with open("SPACE_README_PHI2_DREAM.md", "w") as f:
    f.write(readme_content)

print("Uploading files to Space...")
try:
    api.upload_file(
        path_or_fileobj="SPACE_README_PHI2_DREAM.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="Dockerfile_phi2_dream",
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
        path_or_fileobj="src/progressive_llm_cognitive_phi2.py",
        path_in_repo="progressive_llm_cognitive_phi2.py",
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
    api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value=f"{USERNAME}/progressive-cognitive-phi2-dream-lora")
    print("Secrets added successfully.")
except Exception as e:
    print(f"Error adding secrets: {e}")

print(f"\nDone! Training will run on T4 GPU.")
print(f"Logs: https://huggingface.co/spaces/{REPO_ID}?logs=container")
print(f"Model will be saved to: {USERNAME}/progressive-cognitive-phi2-dream-lora")
