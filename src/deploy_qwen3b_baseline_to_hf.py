"""
Deploy Qwen 3B Baseline (Flat-LoRA) Training to HF Spaces.
Control group for cross-architecture validation.
"""

import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "progressive-cognitive-qwen3b-baseline"
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

COPY baseline_training_qwen3b.py .

CMD python -m http.server 7860 & python baseline_training_qwen3b.py
"""

readme_content = """---
title: Qwen 3B Baseline Training
emoji: 📊
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---

# Qwen 2.5 3B — Baseline (Flat-LoRA)

Control group: same data, flat training, no curriculum, no SVD.
"""

for fname, content in [
    ("Dockerfile_q3b_base", dockerfile_content),
    ("README_q3b_base.md", readme_content),
]:
    with open(fname, "w") as f:
        f.write(content)

print("Uploading files...")
try:
    api.upload_file(path_or_fileobj="README_q3b_base.md", path_in_repo="README.md", repo_id=REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj="Dockerfile_q3b_base", path_in_repo="Dockerfile", repo_id=REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj="requirements.txt", path_in_repo="requirements.txt", repo_id=REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj="src/baseline_training_qwen3b.py", path_in_repo="baseline_training_qwen3b.py", repo_id=REPO_ID, repo_type="space")
    print("Files uploaded.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print("Setting secrets...")
api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=TOKEN)
api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value="dexmac/progressive-cognitive-qwen3b-baseline-lora")

print("Requesting T4 GPU...")
api.request_space_hardware(repo_id=REPO_ID, hardware="t4-small")

print(f"\n✅ Qwen 3B Baseline training deployed!")
print(f"   URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"   Logs: https://huggingface.co/spaces/{REPO_ID}?logs=container")

for f in ["Dockerfile_q3b_base", "README_q3b_base.md"]:
    if os.path.exists(f):
        os.remove(f)
