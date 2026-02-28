"""
Deploy English Evaluation to HuggingFace Spaces.
Tests all 6 models: Qwen 1.5B & 3B × (Base, Flat-LoRA, Dream)
"""
import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN")
assert TOKEN, "Set HF_TOKEN environment variable"
SPACE_NAME = "progressive-cognitive-eval-en"
USERNAME = "dexmac"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

api = HfApi(token=TOKEN)

# ─── Create Space ───
print(f"Creating Space {REPO_ID}...")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print("Space created/found successfully.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# ─── Dockerfile ───
dockerfile = """FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV OMP_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluation_framework.py .
COPY run_english_evaluation.py .

# HTTP healthcheck on 7860 + run evaluation
CMD python -m http.server 7860 & python run_english_evaluation.py
"""

# ─── requirements.txt ───
requirements = """torch>=2.0.0
transformers>=4.38.0
peft>=0.8.0
accelerate>=0.26.0
huggingface_hub>=0.20.0
sentencepiece
protobuf
"""

# ─── README ───
readme = """---
title: Progressive Cognitive Evaluation (English)
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Progressive Cognitive Architecture — English Evaluation

Comparative evaluation of **6 models** across **5 dimensions**:

| Model | Type |
|-------|------|
| Qwen2.5-1.5B (Base) | No fine-tuning |
| Qwen2.5-1.5B + Flat LoRA (EN) | Standard LoRA baseline |
| Qwen2.5-1.5B + Dream (EN) | Progressive 4-phase + Dream Pruning |
| Qwen2.5-3B (Base) | No fine-tuning |
| Qwen2.5-3B + Flat LoRA (EN) | Standard LoRA baseline |
| Qwen2.5-3B + Dream (EN) | Progressive 4-phase + Dream Pruning |

**Test dimensions:**
1. Exact accuracy
2. Number sense (intuition)
3. Self-awareness (delegation)
4. Adversarial robustness
5. Error patterns analysis

Results are saved to `dexmac/progressive-cognitive-results` dataset.
"""

# ─── Upload files ───
print("Uploading files...")

files_to_upload = [
    ("Dockerfile", dockerfile),
    ("README.md", readme),
    ("requirements.txt", requirements),
]

for filename, content in files_to_upload:
    api.upload_file(
        path_or_fileobj=content.encode(),
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"  ✅ {filename}")

# Upload actual Python files
for local_path, repo_path in [
    ("src/evaluation_framework.py", "evaluation_framework.py"),
    ("src/run_english_evaluation.py", "run_english_evaluation.py"),
]:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="space",
    )
    print(f"  ✅ {repo_path}")

# ─── Secrets ───
print("Adding secrets...")
api.add_space_secret(repo_id=REPO_ID, key="HF_TOKEN", value=TOKEN)
api.add_space_secret(repo_id=REPO_ID, key="HF_REPO_ID", value="dexmac/progressive-cognitive-results")
print("  ✅ Secrets set")

# ─── Request GPU ───
print("Requesting T4 GPU...")
try:
    api.request_space_hardware(repo_id=REPO_ID, hardware="t4-small")
    print("  ✅ T4 GPU requested")
except Exception as e:
    print(f"  ⚠ Error: {e}")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Evaluation Space deployed!                                   ║
║  URL: https://huggingface.co/spaces/{REPO_ID}  ║
║  Logs: https://huggingface.co/spaces/{REPO_ID}?logs=container ║
║                                                               ║
║  6 models × 5 tests × 50 samples = ~30-45 min estimated     ║
║  Auto-pauses when done. Results → HF dataset.                ║
╚══════════════════════════════════════════════════════════════╝
""")
