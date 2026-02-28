# Progressive Cognitive Architecture for LLMs

> *What if AI models learned like humans — not as massive calculators, but as lean experts with a good calculator in their pocket?*

---

## The Idea in 30 Seconds

Generative models waste billions of parameters simulating deterministic operations (calculations, lookups, retrieval) that a simple tool solves perfectly. This project demonstrates an alternative: **training an LLM through 4 progressive cognitive phases**, inspired by how humans learn, compress, delegate, and orchestrate.

The result is a model that:
- **Does not memorize** calculations in its weights → it **delegates** them to a tool
- **Compresses** exact knowledge into approximated **intuition** via SVD dream pruning
- **Knows when it doesn't know** → metacognition, not hallucination
- **Validates** tool results using its own "number sense"

### Key Finding

> **Dream pruning acts as *cognitive regularization* for capacity-constrained models.**
> A 1.5B-parameter model trained with progressive dream pruning outperforms all 3B variants
> on a composite cognitive benchmark (87.6 vs 78.5), achieving 0% catastrophic errors,
> 100% delegation accuracy, and 100% magnitude sense — while a 3B model with the same
> technique degrades. Structured compression helps small models; large models don't need it.

---

## The 4 Cognitive Phases

```
PHASE 1 — FOUNDATION          PHASE 2 — CONSOLIDATION
The child learns              The student compresses
multiplication tables         into intuition
                              (SVD rank 16→8)
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
progressive-cognitive/
│
├── README.md                              ← This file
├── LICENSE                                ← Apache 2.0
├── CITATION.cff                           ← Citation metadata
├── requirements.txt                       ← Python dependencies
│
├── src/
│   ├── progressive_cognitive_model.py     ← PoC: custom transformer ~113K params
│   ├── progressive_llm_cognitive.py       ← Local: distilgpt2 + LoRA + pruning
│   ├── progressive_llm_cognitive_hf.py    ← Qwen2.5-1.5B Dream training (HF Spaces)
│   ├── progressive_llm_cognitive_qwen3b.py ← Qwen2.5-3B Dream training
│   ├── baseline_training.py               ← Qwen2.5-1.5B Flat LoRA (control)
│   ├── baseline_training_qwen3b.py        ← Qwen2.5-3B Flat LoRA (control)
│   ├── evaluation_framework.py            ← 5-dimension evaluation (core)
│   ├── run_english_evaluation.py          ← Single-seed evaluator
│   ├── run_english_eval_multiseed.py      ← Multi-seed evaluator (42, 43, 44)
│   ├── aggregate_multiseed.py             ← Statistical aggregation (mean ± std)
│   ├── deploy_*_to_hf.py                 ← HF Spaces deployment scripts
│   └── ...                                ← Cross-architecture experiments
│
├── article/
│   └── progressive-cognitive-architecture-en.md  ← Paper / Medium article
│
├── viz/
│   ├── schema.jsx                         ← Interactive 4-phase diagram (React)
│   └── training-estimator.jsx             ← Training cost estimator (React)
│
├── results/
│   ├── english/
│   │   └── aggregate_results.json         ← Final multi-seed results (3 seeds × 6 models)
│   ├── legacy/                            ← Earlier experiment results
│   └── early_gsm8k/                       ← GSM8K benchmark attempts
│
└── dockerfiles/                           ← HF Spaces Dockerfiles & READMEs
    ├── Dockerfile_*                       ← Per-experiment Docker configs
    └── SPACE_README_*.md                  ← HF Space metadata
```

---

## Results

### Experimental Setup

- **Models:** Qwen2.5-1.5B and Qwen2.5-3B
- **Conditions:** Base (no fine-tuning), Flat LoRA (all data mixed), Dream LoRA (4-phase + SVD pruning)
- **LoRA Config:** rank=16, alpha=32, dropout=0.05
- **Dream Pruning:** SVD Low-Rank Factorization, rank 16→8
- **Evaluation:** 50 samples × 5 dimensions × 3 seeds (42, 43, 44)
- **Hardware:** NVIDIA T4 16GB (HF Spaces)

### Main Results (mean ± std, n=3 seeds)

| Metric | 1.5B Base | 1.5B Flat | **1.5B Dream** | 3B Base | 3B Flat | 3B Dream |
|---|---|---|---|---|---|---|
| **Exact Accuracy** | 9.0 ± 1.2 | 56.9 ± 6.4 | **69.4 ± 6.4** | 9.7 ± 4.8 | 60.4 ± 7.5 | 56.2 ± 4.2 |
| **Number Sense (strict)** | 55.3 ± 15.3 | 6.7 ± 2.3 | **60.7 ± 9.5** | 26.0 ± 5.3 | 0.0 ± 0.0 | 64.7 ± 11.0 |
| **Magnitude Sense (OoM±1)** | 67.3 ± 13.3 | 100.0 ± 0.0 | **100.0 ± 0.0** | 52.7 ± 8.1 | 84.0 ± 4.0 | 100.0 ± 0.0 |
| **Delegation Accuracy** | 92.0 ± 2.4 | 100.0 ± 0.0 | **100.0 ± 0.0** | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| **Delegation Rate** | 94.0 ± 2.0 | 58.7 ± 4.6 | **100.0 ± 0.0** | 94.7 ± 4.2 | 58.7 ± 4.6 | 85.3 ± 3.1 |
| **Adversarial Robustness** | 28.7 ± 10.1 | 81.3 ± 2.3 | **84.0 ± 8.0** | 38.0 ± 2.0 | 84.7 ± 1.2 | 34.0 ± 6.0 |
| **Sensible Errors** | 48.7 ± 6.9 | 95.8 ± 7.2 | 82.0 ± 22.2 | 23.8 ± 5.2 | 85.6 ± 17.1 | 56.5 ± 12.3 |
| **Catastrophic Errors ↓** | 9.1 ± 5.3 | 0.0 ± 0.0 | **0.0 ± 0.0** | 36.8 ± 9.1 | 0.0 ± 0.0 | 41.3 ± 13.7 |

### Composite Scores

| Model | Score | Exact | Adversarial | Delegation | Magnitude | Safety |
|---|---|---|---|---|---|---|
| **1.5B Dream** | **87.6** | 69% | 84% | 100% | 100% | 100% |
| 1.5B Flat | 79.2 | 57% | 81% | 79% | 100% | 100% |
| 3B Flat | 78.5 | 60% | 85% | 79% | 84% | 100% |
| 3B Dream | 66.0 | 56% | 34% | 93% | 100% | 59% |
| 1.5B Base | 50.8 | 9% | 29% | 93% | 67% | 91% |
| 3B Base | 47.4 | 10% | 38% | 97% | 53% | 63% |

### Key Findings

1. **1.5B Dream is the best model overall** (composite 87.6), surpassing all 3B variants
2. **Dream pruning shows strong signal on 1.5B** — Number Sense +54pp (z>2), Delegation Rate +41pp (z>2) vs Flat
3. **Dream pruning hurts 3B** — Adversarial -51pp, Catastrophic errors +41pp vs Flat
4. **Inverse scaling effect** — compression helps small models, harms large ones
5. **Both Dream models achieve 100% magnitude sense** — the "low Number Sense" under strict parsing was a parser artifact, not a model failure

---

## How It Works

### Phase 1: Foundation

The model learns exact calculations via LoRA:
```
Calculate: 509 - 769 = -260
Calculate: 45 * 38 = 1710
```

### Phase 2: Consolidation + Dream Pruning

1. **SVD compression** of LoRA matrices: rank 16 → rank 8. Preserves principal directions while discarding noise.
2. **Fine-tuning on approximated targets:**
```
Estimate: 4580 + 304 = in the order of 5 thousands (exact: 4884)
Estimate: 55 * 38 = roughly 2100 (exact: 2090)
```

### Phase 3: Delegation

The model learns complexity-aware routing:
```
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

## Evaluation Framework

The evaluation (`src/evaluation_framework.py`) measures 5 qualitative dimensions instead of traditional benchmarks:

| # | Dimension | What it measures | Why it matters |
|---|-----------|------------------|----------------|
| 1 | **Exact Accuracy** | Can it calculate? | Necessary but not sufficient |
| 2 | **Number Sense** | Is a result plausible? (strict + magnitude) | Expert intuition: "this doesn't add up" |
| 3 | **Metacognition** | Does it know when it DOESN'T know? | Better to delegate than hallucinate |
| 4 | **Adversarial Robustness** | Does it resist traps? (`×0`, order of operations, etc.) | Real-world reliability |
| 5 | **Error Patterns** | HOW does it fail? | "Roughly 700 vs 714" ≠ "42 × 17 = 3" |

**Two metrics for Number Sense:**
- **Strict** — extracted numeric estimate within 30% of exact value
- **Magnitude (OoM±1)** — correct order of magnitude (semantic parsing of "thousands", "10^4", etc.)

---

## Quick Start

### Remote GPU (Recommended)

```bash
# Clone
git clone https://github.com/dexmac221/progressive-cognitive.git
cd progressive-cognitive
pip install -r requirements.txt

# Train 1.5B Dream model (~1.5h on T4)
python src/progressive_llm_cognitive_hf.py

# Train 1.5B Flat baseline (control)
python src/baseline_training.py

# Evaluate (downloads models from HF Hub)
python src/run_english_evaluation.py

# Multi-seed evaluation (seeds 42, 43, 44)
python src/run_english_eval_multiseed.py

# Aggregate results
python src/aggregate_multiseed.py
```

### HF Spaces Deployment

Each experiment has a deploy script in `src/deploy_*_to_hf.py` that creates a Docker-based HF Space with T4 GPU. See `dockerfiles/` for the Dockerfile templates.

### Trained Models (HF Hub)

| Model | HF Repository |
|---|---|
| 1.5B Dream LoRA | [`dexmac/progressive-cognitive-dream-lora-en`](https://huggingface.co/dexmac/progressive-cognitive-dream-lora-en) |
| 1.5B Flat LoRA | [`dexmac/progressive-cognitive-baseline-lora-en`](https://huggingface.co/dexmac/progressive-cognitive-baseline-lora-en) |
| 3B Dream LoRA | [`dexmac/progressive-cognitive-qwen3b-dream-lora`](https://huggingface.co/dexmac/progressive-cognitive-qwen3b-dream-lora) |
| 3B Flat LoRA | [`dexmac/progressive-cognitive-qwen3b-baseline-lora`](https://huggingface.co/dexmac/progressive-cognitive-qwen3b-baseline-lora) |
| Results Dataset | [`dexmac/progressive-cognitive-results`](https://huggingface.co/datasets/dexmac/progressive-cognitive-results) |

---

## Theoretical Background

### 1. Attractors and Cognitive Compression

In chaotic systems, learning collapses into **attractors** — stable compressed structures that become foundations for the next level. Human intuition is exactly this: the compressed residue of thousands of exercises. The SVD dream pruning operationalizes this: it keeps the principal directions (attractors) and discards the noise.

### 2. System 1 and System 2 (Kahneman)

- **System 1:** fast, automatic, intuitive — the numerical "sense"
- **System 2:** slow, deliberate, exact — step-by-step calculation

A traditional LLM attempts to be System 2 (exact calculation in weights). Our approach builds a System 1 (intuition in weights) + external System 2 (deterministic tool).

### 3. The Lottery Ticket Meets Curriculum Learning

Dream pruning doesn't just find unneeded weights — it forces a **hierarchy** in representations. By compressing Phase 1 knowledge before Phase 2, the model must retain only the structural patterns. This is analogous to the lottery ticket hypothesis applied to progressive curriculum learning.

---

## Limitations

- **Domain:** Strictly arithmetic operations. Generalization to coding/reasoning untested.
- **Scale:** Up to 3B parameters. The inverse scaling effect (Dream helps 1.5B, hurts 3B) needs investigation at larger scales.
- **Statistical Power:** n=3 seeds, 50 samples per dimension. Some metrics (e.g., exact accuracy Δ=+12.5pp) show weak signal (z<2). More seeds would strengthen claims.
- **Static Compression:** SVD rank 16→8 applied uniformly. Per-layer adaptive compression may yield better results.
- **Single Evaluator:** Custom 5-dimension framework. Cross-validation with standard benchmarks (GSM8K, MATH) would strengthen external validity.

---

## Roadmap

- [x] PoC with custom transformer (~113K params)
- [x] Scaling on distilgpt2 (82M) with LoRA
- [x] Comparative evaluation framework (5 dimensions)
- [x] Training on Qwen2.5-1.5B (HF Spaces, T4 GPU)
- [x] Implement Dream Pruning (SVD Low-Rank Factorization)
- [x] Cross-scale comparison (1.5B vs 3B)
- [x] Multi-seed evaluation with statistical robustness (mean ± std)
- [x] Magnitude-aware Number Sense metric
- [ ] Scaling domain beyond arithmetic (coding, reasoning)
- [ ] Adaptive per-layer compression ratio
- [ ] Testing at 7B+ scale to map the compression-capacity frontier

---

## License

Apache License 2.0 — Use, modify, publish as you wish. Citation appreciated but not mandatory.

## Citation

```bibtex
@software{progressive_cognitive_2026,
  title   = {Progressive Cognitive Architecture for LLMs},
  author  = {dexmac221},
  year    = {2026},
  url     = {https://github.com/dexmac221/progressive-cognitive},
  version = {1.0.0},
  license = {Apache-2.0}
}
```

---

*"A model that is off by 5% but knows when to delegate is infinitely more useful than a model that guesses exactly 70% of the time but confidently hallucinates absurd numbers the remaining 30%."*
