import os
os.environ["OMP_NUM_THREADS"] = "1"

"""
╔══════════════════════════════════════════════════════════════════╗
║   ENGLISH EVALUATION — All 6 Models                             ║
║                                                                  ║
║   Tests: Qwen 1.5B & 3B × (Base, Flat-LoRA, Dream)             ║
║   One model at a time to fit T4 16GB VRAM                       ║
║   Results saved to HF Hub + auto-pause                          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import gc
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download, HfApi
from evaluation_framework import (
    TestConfig, TestSuiteGenerator, ModelEvaluator,
    ComparativeReport, ResponseAnalyzer
)


# ─────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "Qwen2.5-1.5B (Base)",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "lora_repo": None,
        "lora_subfolder": None,
    },
    {
        "name": "Qwen2.5-1.5B + Flat LoRA (EN)",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "lora_repo": "dexmac/progressive-cognitive-baseline-lora-en",
        "lora_subfolder": None,  # adapters at root
    },
    {
        "name": "Qwen2.5-1.5B + Dream (EN)",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "lora_repo": "dexmac/progressive-cognitive-dream-lora-en",
        "lora_subfolder": "lora_adapters",  # adapters in subfolder
    },
    {
        "name": "Qwen2.5-3B (Base)",
        "base_model": "Qwen/Qwen2.5-3B",
        "lora_repo": None,
        "lora_subfolder": None,
    },
    {
        "name": "Qwen2.5-3B + Flat LoRA (EN)",
        "base_model": "Qwen/Qwen2.5-3B",
        "lora_repo": "dexmac/progressive-cognitive-qwen3b-baseline-lora",
        "lora_subfolder": None,
    },
    {
        "name": "Qwen2.5-3B + Dream (EN)",
        "base_model": "Qwen/Qwen2.5-3B",
        "lora_repo": "dexmac/progressive-cognitive-qwen3b-dream-lora",
        "lora_subfolder": "lora_adapters",
    },
]


# ─────────────────────────────────────────────────────────────────
# REAL MODEL EVALUATOR
# ─────────────────────────────────────────────────────────────────

class RealModelEvaluator(ModelEvaluator):
    """Evaluator that uses a real HuggingFace model for inference."""

    def __init__(self, name, config, model, tokenizer):
        super().__init__(name, config)
        self.model = model
        self.tokenizer = tokenizer

    def _simulate_response(self, prompt):
        """Override simulation with real model inference."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()


# ─────────────────────────────────────────────────────────────────
# LOAD & EVALUATE ONE MODEL AT A TIME
# ─────────────────────────────────────────────────────────────────

def free_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_and_evaluate(model_def, test_suite, config):
    """Load a single model, evaluate it, and unload it."""
    name = model_def["name"]
    base_id = model_def["base_model"]
    lora_repo = model_def["lora_repo"]
    lora_subfolder = model_def["lora_subfolder"]

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {name}")
    print(f"{'='*60}")

    device = config.device
    start_time = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"  Loading base model: {base_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Apply LoRA if specified
    if lora_repo:
        print(f"  Downloading LoRA from: {lora_repo}...")
        try:
            repo_path = snapshot_download(repo_id=lora_repo)
            lora_path = (
                os.path.join(repo_path, lora_subfolder)
                if lora_subfolder
                else repo_path
            )
            print(f"  Loading LoRA adapters from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print(f"  LoRA merged successfully.")
        except Exception as e:
            print(f"  ⚠ Error loading LoRA: {e}")
            print(f"  Continuing with base model only.")

    model.eval()
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.1f}s")

    # Check VRAM
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {mem_used:.1f}GB / {mem_total:.1f}GB")

    # Evaluate
    evaluator = RealModelEvaluator(name, config, model, tokenizer)
    results = evaluator.evaluate_suite(test_suite)

    eval_time = time.time() - start_time - load_time
    print(f"  Evaluation completed in {eval_time:.1f}s")

    # Unload model
    del model
    del tokenizer
    del evaluator
    free_gpu_memory()

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"  VRAM after cleanup: {mem_after:.2f}GB")

    return results


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   ENGLISH EVALUATION — Progressive Cognitive Architecture    ║
║   6 Models × 5 Test Dimensions                               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Config — 50 samples per category = 250 tests per model, 1500 total
    config = TestConfig(
        n_samples_per_test=50,
        device=device,
        max_new_tokens=100,
        temperature=0.3,
    )

    # Generate test suite (same for all models — fair comparison)
    print("\n  Generating test suite...")
    suite = TestSuiteGenerator.generate_all(n=config.n_samples_per_test, seed=config.seed)
    total_tests = sum(len(s['tests']) for s in suite.values())
    print(f"  Generated: {total_tests} tests in {len(suite)} categories")

    # Evaluate each model sequentially (one at a time for VRAM)
    all_evaluations = {}
    total_start = time.time()

    for model_def in MODELS:
        try:
            results = load_and_evaluate(model_def, suite, config)
            all_evaluations[model_def["name"]] = results
        except Exception as e:
            print(f"\n  ⚠ FAILED: {model_def['name']}: {e}")
            import traceback
            traceback.print_exc()
            free_gpu_memory()
            continue

    total_time = time.time() - total_start
    print(f"\n  Total evaluation time: {total_time/60:.1f} minutes")

    if not all_evaluations:
        print("  ⚠ No models evaluated successfully!")
        return

    # ─── Generate Comparative Report ───
    report = ComparativeReport(all_evaluations)
    report.generate()

    # Save results
    os.makedirs('./results', exist_ok=True)
    report.save('./results/english_evaluation_report.json')

    # Also save the full raw summary for analysis
    summary = {}
    for model_name, evals in all_evaluations.items():
        summary[model_name] = {}
        for cat, data in evals.items():
            summary[model_name][cat] = data['summary']

    with open('./results/english_evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("  Raw summary saved: results/english_evaluation_summary.json")

    # ─── Push to HF Hub ───
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-results")

    if hf_token:
        print(f"\n  Pushing results to HF Hub: {repo_id}")
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

            api.upload_file(
                path_or_fileobj='./results/english_evaluation_report.json',
                path_in_repo="english_evaluation_report.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            api.upload_file(
                path_or_fileobj='./results/english_evaluation_summary.json',
                path_in_repo="english_evaluation_summary.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            print("  ✅ Results pushed to Hub!")
        except Exception as e:
            print(f"  ⚠ Error pushing to Hub: {e}")

    # ─── Pause the Space ───
    space_id = os.environ.get("SPACE_ID")
    if space_id and hf_token:
        try:
            api = HfApi(token=hf_token)
            print(f"  Pausing Space {space_id} to save credits...")
            api.pause_space(repo_id=space_id)
        except Exception as e:
            print(f"  ⚠ Error pausing Space: {e}")

    print("\n  ✅ All done!")


if __name__ == "__main__":
    main()
