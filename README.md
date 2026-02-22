# Progressive Cognitive Architecture for LLMs

> *What if AI models learned like humans?*
> *Not massive calculators, but lean experts with a good calculator in their pocket.*

---

## The Idea in 30 Seconds

Generative models waste billions of parameters simulating deterministic operations (calculations, lookups, retrieval) that a simple tool solves perfectly. This project demonstrates an alternative approach: **training an LLM through 4 progressive cognitive phases**, much like a human who learns, compresses, delegates, and orchestrates.

The result is a model that:
- **Does not memorize** calculations in its weights → it **delegates** them to a tool
- **Compresses** exact knowledge into approximated **intuition**
- **Knows when it doesn't know** → metacognition, not hallucination
- **Validates** tool results using its own "number sense"

---

## The 4 Cognitive Phases

```
PHASE 1 — FOUNDATION          PHASE 2 — CONSOLIDATION
The child learns              The student compresses
multiplication tables         into intuition
                              (30% pruning)
    ┌─────────┐                   ┌─────────┐
    │  Exact  │                   │ Number  │
    │  Math   │ ──────────────▶   │  Sense  │
    │in weights│                  │in weights│
    └─────────┘                   └─────────┘
                                       │
                                       ▼
PHASE 4 — ORCHESTRATION       PHASE 3 — DELEGATION
The expert who                The adult uses
spots the bug                 the calculator
    ┌─────────┐                   ┌─────────┐
    │ Intuits │                   │Internal/│
    │Delegates│ ◀──────────────   │  Tool   │
    │Validates│                   │ Routing │
    └─────────┘                   └─────────┘
```

**Guiding Principle:** *Knowledge doesn't disappear — it collapses into attractors. Intuition is the compressed residue of experience.*

---

## Project Structure

```
progressive-cognitive-architecture/
│
├── README.md                              ← This file
│
├── src/
│   ├── progressive_cognitive_model.py     ← PoC: custom transformer ~113K params
│   ├── progressive_llm_cognitive.py       ← Main: distilgpt2/TinyLlama + LoRA + pruning
│   ├── progressive_llm_cognitive_hf.py    ← Main: Qwen2.5-1.5B + LoRA (Hugging Face Spaces)
│   └── evaluation_framework.py            ← Comparative evaluation framework (5 dimensions)
│
├── article/
│   └── progressive-cognitive-architecture-en.md ← Medium Article
│
├── viz/
│   ├── schema.jsx                         ← Interactive visualization of the 4 phases (React)
│   └── training-estimator.jsx             ← Training time/cost estimator (React)
│
├── results/                               ← Generated from training
│   ├── metrics.json                       ← PoC metrics (custom transformer)
│   ├── llm_metrics.json                   ← distilgpt2 metrics
│   ├── comparative_report.json            ← Comparative evaluation report
│   ├── qwen_cognitive_report.json         ← Qwen2.5-1.5B evaluation report
│   └── cognitive_model.pt                 ← PoC checkpoint (custom transformer)
│
└── requirements.txt                       ← Python dependencies
```

---

## Quick Start — Remote GPU Training

### 1. Requirements

```bash
pip install torch transformers peft accelerate datasets
```

**Minimum Hardware:** 1× GPU with ≥16GB VRAM (T4 or A10G)
**Recommended Hardware:** 1× A100 80GB or H100

### 2. Training with Qwen2.5-1.5B (Full Experiment)

To scale on Qwen2.5, modify `progressive_llm_cognitive_hf.py`:

```python
class Config:
    model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda:0"
    
    # LoRA — higher rank for larger models
    lora_r = 16
    lora_alpha = 32
    
    # Scaled Training
    batch_size = 32
    max_seq_len = 256
    
    phase1_epochs = 8
    phase1_lr = 3e-4
    phase1_samples = 50000
    
    phase2_epochs = 6
    phase2_lr = 2e-4
    phase2_samples = 30000
    
    phase3_epochs = 6
    phase3_lr = 1e-4
    phase3_samples = 30000
    
    phase4_epochs = 4
    phase4_lr = 5e-5
    phase4_samples = 20000
```

**Estimated Time:** ~1.5 hours on 1× T4 GPU (Hugging Face Spaces)

### 3. Multi-GPU (Optional)

To use 2 GPUs with `accelerate`:

```bash
accelerate config  # interactive setup
accelerate launch src/progressive_llm_cognitive_hf.py
```

### 4. Comparative Evaluation

After training, run the evaluation framework:

```bash
python src/evaluation_framework.py
```

This compares 3 approaches across 5 dimensions (see [Evaluation](#evaluation) section).

---

## How It Works — Technical Details

### Phase 1: Foundation

The model learns exact calculations via LoRA. Data format:

```
Calculate: 509 - 769 = -260
Calculate: 45 * 38 = 1710
```

**LoRA rank 16 on Qwen2.5-1.5B** → ~11M trainable parameters out of 1.5B total (<1%).

### Phase 2: Consolidation (The Most Significant Data Point)

1. **Structured Pruning** — Removal of 30% of LoRA weights by magnitude
2. **Fine-tuning on Approximated Targets** — No longer exact answers, but estimates:

```
Estimate: 4580 + 304 = in the order of 5 thousands (exact: 4884)
Estimate: 55 * 38 = roughly 2100 (exact: 2090)
```

**Key Result:** The loss after pruning is lower than before. Fewer parameters + approximated targets = better learning. This empirically confirms the hypothesis: a lean model that approximates is more efficient than a full one attempting exactness.

### Phase 3: Delegation

The model learns to discriminate complexity and decide routing:

```
Analyze: 5 + 3
Complexity: elementary
Decision: INTERNAL CALCULATION

Analyze: 847 * 93
Complexity: complex
Decision: DELEGATE TO TOOL
Reason: calculation too complex for reliable estimation
Tool result: 78771
```

### Phase 4: Orchestration

Full pipeline — intuition, routing, tool, validation:

```
Solve: 342 * 67
Step 1 - Intuition: in the order of tens of thousands
Step 2 - Routing: DELEGATE (medium complexity)
Step 3 - Tool: 22914
Step 4 - Validation: result 22914 consistent with estimate → VALID
```

---

## Evaluation

The evaluation framework (`evaluation_framework.py`) does not use traditional benchmarks. It measures 5 qualitative dimensions:

### The 5 Dimensions

| # | Test | What it measures | Why it matters |
|---|------|------------------|----------------|
| 1 | **Exact Accuracy** | Can it calculate? | The classic benchmark — necessary but not sufficient |
| 2 | **Number Sense** | Is a result plausible? | The expert's intuition: "this number doesn't add up" |
| 3 | **Metacognition** | Does it know when it DOESN'T know? | Better to delegate than hallucinate |
| 4 | **Robustness** | Does it resist traps? | `x × 0`, order of operations, carryovers, negative numbers |
| 5 | **Error Patterns** | HOW does it fail? | An error "roughly 700 vs 714" ≠ an error "42 × 17 = 3" |

### The 3 Compared Models

| Model | Description |
|-------|-------------|
| **Baseline** | Qwen2.5-1.5B base (no fine-tuning) |
| **Progressive** | Qwen2.5-1.5B + LoRA 4 cognitive phases + 30% pruning |
| **Tool-only** | Qwen2.5-1.5B base + deterministic calculator |

### Error Classification

The framework classifies each error qualitatively:

- `correct` — exact answer
- `close_estimate` — error <10%, good intuition
- `rough_estimate` — error 10-50%, rough estimate but not absurd
- `same_magnitude_wrong` — correct order of magnitude, wrong value
- `sign_error` — correct magnitude, wrong sign
- `magnitude_off_by_one` — wrong by one order of magnitude
- `magnitude_catastrophic` — wrong by 2+ orders of magnitude
- `no_answer` — no extractable number

**The Key Insight:** Two models with the same accuracy can have qualitatively different errors. A model with 50% accuracy that only makes "sensible" errors is more useful than one with 60% accuracy that makes catastrophic errors the remaining 40% of the time.

---

## Real Results (Qwen2.5-1.5B)

We tested the progressive cognitive architecture on **Qwen2.5-1.5B**, comparing the base model (without fine-tuning) with the model trained through the 4 cognitive phases (with merged LoRA weights). The results unequivocally demonstrate the effectiveness of the approach.

### Direct Comparison (Base vs Progressive)

| Cognitive Dimension | Qwen2.5-1.5B (Base) | Qwen2.5-1.5B (Progressive) | Variation | Insight |
|---------------------|---------------------|----------------------------|-----------|---------|
| **1. Exact Calculation** | 22.2% | **33.3%** | +11.1% | Training improves pure calculation, even if it's not the primary goal. |
| **2. Number Sense** | 60.0% | **60.0%** | = | No *catastrophic forgetting*. Basic intuition is preserved. |
| **3. Metacognition** | Correct Delegation: 90.9%<br>Delegation Rate: 80.0% | **Correct Delegation: 100%**<br>**Delegation Rate: 100%** | **Perfect** | **Clear victory.** The model learns not to hallucinate: when the calculation is complex, it *always* and *correctly* delegates to the tool. |
| **4. Robustness** | 25.0% (6 severe errors) | **60.0%** (0 severe errors) | **+35.0%** | **The most impressive result.** The base model falls into math traps. The progressive model resists and, if it fails, only makes "sensible" errors. |

### Error Pattern Analysis

It's not just about *how much* the model fails, but *how* it fails.
- **Base Model**: Commits catastrophic errors (wrong orders of magnitude, invents numbers).
- **Progressive Model**: **100%** of errors in robustness tests are classified as "sensible" (e.g., close estimates, same order of magnitude). The model has developed a true mathematical intuition that prevents it from giving absurd answers.

---

## Roadmap

- [x] PoC with custom transformer (~113K params)
- [x] Scaling on distilgpt2 (82M) with LoRA
- [x] Comparative evaluation framework (5 dimensions)
- [x] Training on Qwen2.5-1.5B on Hugging Face Spaces (T4 GPU)
- [x] Real comparative evaluation (Base vs Progressive)
- [ ] Scaling the domain beyond arithmetic (coding, reasoning)
- [ ] Paper / article with full results

---

## Theoretical Background

The project is founded on three converging intuitions:

### 1. Attractors and Cognitive Compression

In chaotic systems, every level of learning collapses into an **attractor** — a stable compressed structure that becomes the foundation for the next level. Human intuition is exactly this: the compressed residue of thousands of exercises, errors, and repetitions. You don't remember multiplication tables because you calculate them — you *know* them because they have become structure.

### 2. System 1 and System 2 (Kahneman)

- **System 1:** fast, automatic, intuitive — the numerical "sense"
- **System 2:** slow, deliberate, exact — step-by-step calculation

A traditional LLM attempts to be System 2 (exact calculation in weights). Our approach builds a System 1 (intuition in weights) + external System 2 (deterministic tool).

### 3. The Programmer Analogy

A programmer with 25 years of experience doesn't remember syntax — they use an AI to generate code. But they spot the bug without reading a line, because they have developed a "sense" for software. Operational knowledge has been delegated to the tool; structural intuition remains in the compressed neural circuits.

---

## License

Apache License 2.0 — Use, modify, publish as you wish. If you cite the project, it's appreciated but not mandatory.

---

*"Humans like to measure everything, a virtue but also a flaw. A complex system is understood, not completely measured."*
