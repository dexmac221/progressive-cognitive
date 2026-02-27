"""
Deploy Dream Cycle Experiment to HF Spaces.
Trains 3 models (Dream Cycle, Flat Continuous, Fresh Logic)
and pushes results + models to HF Hub.
"""

import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
SPACE_NAME = "progressive-cognitive-dream-cycle"
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

# ─── Dockerfile ───
dockerfile_content = """
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV OMP_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

COPY dream_cycle_training.py .

CMD python -m http.server 7860 & python dream_cycle_training.py
"""

with open("Dockerfile_dream_cycle", "w") as f:
    f.write(dockerfile_content)

# ─── README ───
readme_content = """---
title: Dream Cycle Experiment
emoji: 🌙
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Dream Cycle Experiment — Continual Learning via SVD

**Hypothesis**: SVD compression (Dream Pruning) acts as a "sleep cycle" 
that consolidates knowledge and prevents catastrophic forgetting.

## Models Trained

| Model | Description |
|-------|-------------|
| **A: Dream Cycle** | Dream-LoRA (arithmetic) → logic + SVD cycles |
| **B: Flat Continuous** | Dream-LoRA (arithmetic) → logic (no SVD) |
| **C: Fresh Logic** | Fresh LoRA → logic only (control) |

## Key Question

Does the Dream Cycle (Model A) preserve arithmetic knowledge while 
learning logic, while Flat Continuous (Model B) forgets it?

If yes → SVD acts like "sleep", consolidating memories.
"""

with open("SPACE_README_DREAM_CYCLE.md", "w") as f:
    f.write(readme_content)

# ─── Requirements ───
requirements_content = """torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
datasets>=2.16.0
huggingface_hub
"""

with open("requirements_dream_cycle.txt", "w") as f:
    f.write(requirements_content)

# ─── Upload files ───
print("Uploading files to Space...")
try:
    api.upload_file(
        path_or_fileobj="SPACE_README_DREAM_CYCLE.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="Dockerfile_dream_cycle",
        path_in_repo="Dockerfile",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="requirements_dream_cycle.txt",
        path_in_repo="requirements.txt",
        repo_id=REPO_ID,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj="src/dream_cycle_training.py",
        path_in_repo="dream_cycle_training.py",
        repo_id=REPO_ID,
        repo_type="space"
    )
    print("Files uploaded successfully.")
except Exception as e:
    print(f"Error uploading files: {e}")
    exit(1)

# ─── Set secrets ───
print("Setting secrets...")
try:
    api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=TOKEN)
    api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value="dexmac/progressive-cognitive-results")
    print("Secrets set.")
except Exception as e:
    print(f"Error setting secrets: {e}")

# ─── Request GPU ───
print("Requesting T4 GPU hardware...")
try:
    api.request_space_hardware(repo_id=REPO_ID, hardware="t4-small")
    print("T4 GPU requested!")
except Exception as e:
    print(f"Error requesting hardware: {e}")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Dream Cycle Space deployed!                                 ║
║  URL: https://huggingface.co/spaces/{REPO_ID}
║  Logs: https://huggingface.co/spaces/{REPO_ID}?logs=container
║                                                              ║
║  The Space will:                                             ║
║  1. Train 3 models (Dream Cycle, Flat, Fresh)                ║
║  2. Test arithmetic retention + logic acquisition            ║
║  3. Push results to dexmac/progressive-cognitive-results     ║
║  4. Push trained models to HF Hub                            ║
║  5. Auto-pause when done                                     ║
╚══════════════════════════════════════════════════════════════╝
""")

# Cleanup temp files
for f in ["Dockerfile_dream_cycle", "SPACE_README_DREAM_CYCLE.md", "requirements_dream_cycle.txt"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Cleaned up {f}")
