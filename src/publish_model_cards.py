"""
Upload professional Model Cards to all 5 progressive-cognitive repos on HF Hub.
Makes models discoverable and downloadable with proper documentation.
"""

import os
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "your_hf_token_here")
api = HfApi(token=TOKEN)

# ─────────────────────────────────────────────────────────────────
# Model Card templates
# ─────────────────────────────────────────────────────────────────

QWEN_DREAM_CARD = """---
language: it
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-1.5B
tags:
  - lora
  - cognitive-architecture
  - progressive-learning
  - dream-pruning
  - svd
  - math
  - arithmetic
  - intuition
  - tool-use
datasets:
  - custom
pipeline_tag: text-generation
---

# Progressive Cognitive Architecture — Dream-LoRA (Qwen 2.5 1.5B)

**The model that develops mathematical intuition through a 4-phase cognitive curriculum + SVD Dream Pruning.**

## 🧠 What is this?

This is a LoRA adapter trained with the **Progressive Cognitive Architecture**, a bio-inspired training methodology that teaches LLMs to develop mathematical intuition rather than memorize answers. The training follows 4 cognitive phases:

1. **Foundations** — Learn exact arithmetic (2,000 examples)
2. **Consolidation** — SVD Dream Pruning compresses exact circuits into intuition (rank 16→8), then fine-tune on approximation (1,500 examples)
3. **Delegation** — Learn when to delegate to a calculator tool vs compute internally (1,500 examples)
4. **Orchestration** — Full pipeline: intuition → routing → tool → validation (1,000 examples)

## 🔬 Dream Pruning (SVD Low-Rank Factorization)

Instead of zeroing out small weights (magnitude pruning), Dream Pruning uses **SVD decomposition** to reduce the effective rank of LoRA matrices from 16 to 8, preserving the principal directions (the "logical connections") while discarding noise. Think of it as the model "sleeping" and consolidating its memories.

## 📊 Results (Ablation Study: 100 tests × 3 seeds)

| Metric | Dream-LoRA | Flat-LoRA | Base |
|--------|-----------|-----------|------|
| Exact Accuracy | 58.6% ± 2.9 | 60.6% ± 3.8 | 18.2% ± 2.9 |
| Number Sense | 60.0% ± 0.8 | 0.0% | 57.0% ± 1.4 |
| Metacognition (delegation) | **100.0%** | 0.0% | 84.9% |
| Sensible Errors | 81.3% | — | — |

**Key insight**: Flat-LoRA wins on raw accuracy but *destroys* number sense and metacognition. Dream-LoRA preserves both while achieving comparable accuracy.

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Load LoRA adapter (note: adapters are in lora_adapters/ subfolder)
model = PeftModel.from_pretrained(base_model, "dexmac/progressive-cognitive-dream-lora", subfolder="lora_adapters")

# Test it
inputs = tokenizer("Calcola: 347 + 891 =", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ⚙️ Training Details

- **Base model**: Qwen/Qwen2.5-1.5B (1.5B parameters, frozen)
- **LoRA config**: rank=16, alpha=32, dropout=0.05, targets=q_proj, k_proj, v_proj, o_proj
- **Dream Pruning**: SVD rank reduction 16→8 via QR+SVD decomposition
- **Training data**: 6,000 synthetic math examples (Italian prompts)
- **Hardware**: NVIDIA T4 (16GB VRAM) on Hugging Face Spaces
- **Training time**: ~45 minutes

## 📄 Paper & Code

- **Article**: [What if AI Models Learned Like Humans Do?](https://medium.com/towards-artificial-intelligence/what-if-ai-models-learned-like-humans-do-c69c19f29d0c)
- **GitHub**: [dexmac221/progressive-cognitive](https://github.com/dexmac221/progressive-cognitive)

## 📜 License

Apache 2.0
"""

QWEN_BASELINE_CARD = """---
language: it
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-1.5B
tags:
  - lora
  - baseline
  - flat-training
  - math
  - arithmetic
  - control-group
datasets:
  - custom
pipeline_tag: text-generation
---

# Flat-LoRA Baseline (Qwen 2.5 1.5B) — Control Group

**Baseline (control group) for the Progressive Cognitive Architecture experiment.**

## What is this?

This is a standard LoRA adapter trained on the **same 6,000 math examples** as the Progressive model, but:
- ❌ No 4-phase curriculum (all data mixed together)
- ❌ No Dream Pruning
- ❌ No progressive complexity

This serves as the **control group** to demonstrate that the improvements come from the cognitive architecture, not simply from LoRA fine-tuning.

## 📊 Results

| Metric | Flat-LoRA (this) | Dream-LoRA | Base |
|--------|-----------------|-----------|------|
| Exact Accuracy | **60.6%** ± 3.8 | 58.6% ± 2.9 | 18.2% |
| Number Sense | **0.0%** ❌ | 60.0% | 57.0% |
| Metacognition | **0.0%** ❌ | 100.0% | 84.9% |

**The Paradox of Accuracy**: Flat-LoRA achieves the highest raw accuracy but completely destroys the model's number sense and ability to delegate. It's an "idiot savant" — good at one thing, bad at everything else.

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

model = PeftModel.from_pretrained(base_model, "dexmac/progressive-cognitive-baseline-lora")

inputs = tokenizer("Calculate: 347 + 891 =", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ⚙️ Training Details

- **Base model**: Qwen/Qwen2.5-1.5B
- **LoRA config**: rank=16, alpha=32, targets=q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training**: 3 epochs, 6,000 mixed samples, lr=1e-4
- **Hardware**: NVIDIA T4

## 📄 Related

- **Progressive model (Dream-LoRA)**: [dexmac/progressive-cognitive-dream-lora](https://huggingface.co/dexmac/progressive-cognitive-dream-lora)
- **GitHub**: [dexmac221/progressive-cognitive](https://github.com/dexmac221/progressive-cognitive)

## 📜 License

Apache 2.0
"""

QWEN_PROGRESSIVE_CARD = """---
language: it
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-1.5B
tags:
  - lora
  - cognitive-architecture
  - progressive-learning
  - magnitude-pruning
  - math
  - arithmetic
datasets:
  - custom
pipeline_tag: text-generation
---

# Progressive Cognitive Architecture — Progressive-LoRA (Qwen 2.5 1.5B)

**4-phase progressive training with magnitude pruning (the predecessor to Dream Pruning).**

## What is this?

This is the original Progressive-LoRA model using **magnitude pruning** (zeroing small weights) instead of SVD Dream Pruning. It was the first version of the progressive cognitive architecture.

## 📊 Results

| Metric | Progressive-LoRA (this) | Dream-LoRA | Flat-LoRA |
|--------|------------------------|-----------|-----------|
| Exact Accuracy | 37.0% ± 0.5 | 58.6% ± 2.9 | 60.6% |
| Number Sense | 57.7% ± 0.5 | 60.0% ± 0.8 | 0.0% |
| Metacognition | 98.5% | 100.0% | 0.0% |

Dream Pruning (SVD) significantly improved upon magnitude pruning by preserving more of the learned information during compression.

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Note: adapters are in lora_adapters/ subfolder
model = PeftModel.from_pretrained(base_model, "dexmac/progressive-cognitive-lora", subfolder="lora_adapters")

inputs = tokenizer("Calcola: 347 + 891 =", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📄 Related

- **Dream-LoRA (improved)**: [dexmac/progressive-cognitive-dream-lora](https://huggingface.co/dexmac/progressive-cognitive-dream-lora)
- **GitHub**: [dexmac221/progressive-cognitive](https://github.com/dexmac221/progressive-cognitive)

## 📜 License

Apache 2.0
"""

PHI2_DREAM_CARD = """---
language: en
license: apache-2.0
library_name: peft
base_model: microsoft/phi-2
tags:
  - lora
  - cognitive-architecture
  - progressive-learning
  - dream-pruning
  - svd
  - math
  - arithmetic
  - intuition
  - tool-use
  - cross-architecture
datasets:
  - custom
pipeline_tag: text-generation
---

# Progressive Cognitive Architecture — Dream-LoRA (Phi-2 2.7B)

**Cross-architecture validation: same Progressive Cognitive method on a completely different model family.**

## 🔬 Why Phi-2?

We originally validated the Progressive Cognitive Architecture on **Qwen 2.5 1.5B** (Alibaba). To prove the method generalizes beyond a single architecture, we replicated the exact same experiment on **Phi-2 2.7B** (Microsoft):

| | Qwen 2.5 1.5B | Phi-2 2.7B (this) |
|---|---|---|
| **Company** | Alibaba | Microsoft |
| **Parameters** | 1.5B | 2.7B |
| **Attention output** | `o_proj` | `dense` |
| **MLP** | SwiGLU (gate/up/down) | Standard (fc1/fc2) |
| **Method** | 4 phases + SVD rank 16→8 | **Identical** |
| **Language** | Italian | English |

## 🧠 The 4 Cognitive Phases

1. **Foundations** — Exact arithmetic (2,000 examples)
2. **Consolidation** — SVD Dream Pruning (rank 16→8) + intuition training (1,500 examples)
3. **Delegation** — When to use calculator vs compute internally (1,500 examples)
4. **Orchestration** — Full pipeline: intuition → routing → tool → validation (1,000 examples)

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Note: adapters are in lora_adapters/ subfolder
model = PeftModel.from_pretrained(base_model, "dexmac/progressive-cognitive-phi2-dream-lora", subfolder="lora_adapters")

inputs = tokenizer("Calculate: 347 + 891 =", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ⚙️ Training Details

- **Base model**: microsoft/phi-2 (2.7B parameters, frozen)
- **LoRA config**: rank=16, alpha=32, dropout=0.05, targets=q_proj, k_proj, v_proj, dense
- **Dream Pruning**: SVD rank reduction 16→8
- **Training data**: 6,000 synthetic math examples (English prompts)
- **Hardware**: NVIDIA T4 (16GB VRAM) on Hugging Face Spaces

## 📄 Related

- **Qwen Dream-LoRA**: [dexmac/progressive-cognitive-dream-lora](https://huggingface.co/dexmac/progressive-cognitive-dream-lora)
- **Phi-2 Baseline**: [dexmac/progressive-cognitive-phi2-baseline-lora](https://huggingface.co/dexmac/progressive-cognitive-phi2-baseline-lora)
- **GitHub**: [dexmac221/progressive-cognitive](https://github.com/dexmac221/progressive-cognitive)

## 📜 License

Apache 2.0
"""

PHI2_BASELINE_CARD = """---
language: en
license: apache-2.0
library_name: peft
base_model: microsoft/phi-2
tags:
  - lora
  - baseline
  - flat-training
  - math
  - arithmetic
  - control-group
  - cross-architecture
datasets:
  - custom
pipeline_tag: text-generation
---

# Flat-LoRA Baseline (Phi-2 2.7B) — Control Group

**Baseline (control group) for the Phi-2 cross-architecture validation.**

## What is this?

Standard LoRA adapter trained on the **same 6,000 math examples** as the Progressive model, but:
- ❌ No 4-phase curriculum (all data mixed)
- ❌ No Dream Pruning
- ❌ No progressive complexity

Compare with [dexmac/progressive-cognitive-phi2-dream-lora](https://huggingface.co/dexmac/progressive-cognitive-phi2-dream-lora) to see the difference the cognitive architecture makes.

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

model = PeftModel.from_pretrained(base_model, "dexmac/progressive-cognitive-phi2-baseline-lora")

inputs = tokenizer("Calculate: 347 + 891 =", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ⚙️ Training Details

- **Base model**: microsoft/phi-2 (2.7B parameters)
- **LoRA config**: rank=16, alpha=32, targets=q_proj, k_proj, v_proj, dense, fc1, fc2
- **Training**: 3 epochs, 6,000 mixed samples, lr=1e-4
- **Hardware**: NVIDIA T4

## 📄 Related

- **Phi-2 Dream-LoRA**: [dexmac/progressive-cognitive-phi2-dream-lora](https://huggingface.co/dexmac/progressive-cognitive-phi2-dream-lora)
- **Qwen Baseline**: [dexmac/progressive-cognitive-baseline-lora](https://huggingface.co/dexmac/progressive-cognitive-baseline-lora)
- **GitHub**: [dexmac221/progressive-cognitive](https://github.com/dexmac221/progressive-cognitive)

## 📜 License

Apache 2.0
"""

# ─────────────────────────────────────────────────────────────────
# Upload all Model Cards
# ─────────────────────────────────────────────────────────────────

cards = {
    "dexmac/progressive-cognitive-dream-lora": QWEN_DREAM_CARD,
    "dexmac/progressive-cognitive-baseline-lora": QWEN_BASELINE_CARD,
    "dexmac/progressive-cognitive-lora": QWEN_PROGRESSIVE_CARD,
    "dexmac/progressive-cognitive-phi2-dream-lora": PHI2_DREAM_CARD,
    "dexmac/progressive-cognitive-phi2-baseline-lora": PHI2_BASELINE_CARD,
}

print("Uploading Model Cards to all repos...\n")

for repo_id, card_content in cards.items():
    try:
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✅ {repo_id}")
    except Exception as e:
        print(f"  ❌ {repo_id}: {e}")

print("\nDone! All models are now publicly available with documentation.")
print("\nYour models:")
for repo_id in cards:
    print(f"  https://huggingface.co/{repo_id}")
