"""
Cross-Architecture Evaluation: Qwen 2.5 1.5B vs Phi-2 2.7B
Compares 6 models on the same English test suite:
  1. Qwen 2.5 1.5B (Base)
  2. Qwen 2.5 1.5B + Flat-LoRA
  3. Qwen 2.5 1.5B + Dream-LoRA
  4. Phi-2 2.7B (Base)
  5. Phi-2 2.7B + Flat-LoRA
  6. Phi-2 2.7B + Dream-LoRA

All tested on identical English prompts for fair comparison.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json
import os
import gc
import random
from collections import defaultdict
from evaluation_framework import TestConfig, TestSuiteGenerator, ModelEvaluator, ComparativeReport


class RealModelEvaluator(ModelEvaluator):
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
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()


def load_model_with_lora(base_model_id, lora_repo_id, lora_subfolder, device, trust_remote_code=True):
    """Load a base model + LoRA adapter from HF Hub."""
    from huggingface_hub import snapshot_download

    print(f"  Loading base: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=trust_remote_code
    )

    if lora_repo_id:
        print(f"  Downloading LoRA: {lora_repo_id} (subfolder={lora_subfolder})")
        try:
            repo_path = snapshot_download(repo_id=lora_repo_id)
            lora_path = os.path.join(repo_path, lora_subfolder) if lora_subfolder else repo_path

            if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
                model = PeftModel.from_pretrained(base_model, lora_path)
                model = model.merge_and_unload()
                print(f"  LoRA loaded and merged successfully!")
            else:
                print(f"  WARNING: adapter_config.json not found at {lora_path}")
                model = base_model
        except Exception as e:
            print(f"  ERROR loading LoRA: {e}")
            model = base_model
    else:
        model = base_model

    model.eval()
    return model


def main():
    print("=" * 70)
    print("  CROSS-ARCHITECTURE EVALUATION")
    print("  Qwen 2.5 1.5B vs Phi-2 2.7B — 6 models, same test suite")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─────────────────────────────────────────────────────────────
    # Model definitions: (name, base_id, lora_repo, lora_subfolder, trust_remote_code)
    # ─────────────────────────────────────────────────────────────
    model_defs = [
        # Qwen family
        ("Qwen 1.5B (Base)",       "Qwen/Qwen2.5-1.5B", None, None, True),
        ("Qwen 1.5B + Flat-LoRA",  "Qwen/Qwen2.5-1.5B", "dexmac/progressive-cognitive-baseline-lora", None, True),
        ("Qwen 1.5B + Dream-LoRA", "Qwen/Qwen2.5-1.5B", "dexmac/progressive-cognitive-dream-lora", "lora_adapters", True),
        # Phi-2 family
        ("Phi-2 2.7B (Base)",       "microsoft/phi-2", None, None, True),
        ("Phi-2 2.7B + Flat-LoRA",  "microsoft/phi-2", "dexmac/progressive-cognitive-phi2-baseline-lora", None, True),
        ("Phi-2 2.7B + Dream-LoRA", "microsoft/phi-2", "dexmac/progressive-cognitive-phi2-dream-lora", "lora_adapters", True),
    ]

    # ─────────────────────────────────────────────────────────────
    # Generate test suite ONCE (English, shared across all models)
    # ─────────────────────────────────────────────────────────────
    config = TestConfig(n_samples_per_test=20, device=device, max_new_tokens=100)
    print("\nGenerating test suite...")
    suite = TestSuiteGenerator.generate_all(n=config.n_samples_per_test, seed=config.seed)
    total_tests = sum(len(s['tests']) for s in suite.values())
    print(f"Generated: {total_tests} tests in {len(suite)} categories\n")

    # ─────────────────────────────────────────────────────────────
    # Evaluate each model sequentially (load, evaluate, unload to save VRAM)
    # ─────────────────────────────────────────────────────────────
    evaluations = {}
    current_base_id = None
    current_tokenizer = None

    for i, (name, base_id, lora_repo, lora_subfolder, trust_rc) in enumerate(model_defs):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(model_defs)}] {name}")
        print(f"{'='*60}")

        # Load tokenizer (reuse if same base model)
        if base_id != current_base_id:
            current_tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust_rc)
            if current_tokenizer.pad_token is None:
                current_tokenizer.pad_token = current_tokenizer.eos_token
            current_base_id = base_id

        # Load model
        model = load_model_with_lora(base_id, lora_repo, lora_subfolder, device, trust_rc)

        # Evaluate
        evaluator = RealModelEvaluator(name, config, model, current_tokenizer)
        evaluations[name] = evaluator.evaluate_suite(suite)

        # Free VRAM
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  VRAM freed. Moving to next model...")

    # ─────────────────────────────────────────────────────────────
    # Generate comparative report
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERATING CROSS-ARCHITECTURE REPORT")
    print("=" * 70)

    report = ComparativeReport(evaluations)
    report.generate()

    os.makedirs('./results', exist_ok=True)
    report.save('./results/cross_architecture_report.json')
    print("\nResults saved to results/cross_architecture_report.json")

    # Also save a human-readable summary
    summary = _build_summary(evaluations)
    with open('./results/cross_architecture_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("Summary saved to results/cross_architecture_summary.json")

    # ─────────────────────────────────────────────────────────────
    # Push results to HF Hub
    # ─────────────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-results")

    if hf_token:
        print(f"\nPushing results to {repo_id}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

            for fname in ['cross_architecture_report.json', 'cross_architecture_summary.json']:
                fpath = f'./results/{fname}'
                if os.path.exists(fpath):
                    api.upload_file(
                        path_or_fileobj=fpath,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token
                    )
            print("Results pushed successfully!")

            # Pause Space
            space_id = os.environ.get("SPACE_ID")
            if space_id:
                print(f"Pausing space {space_id}...")
                api.pause_space(repo_id=space_id)
        except Exception as e:
            print(f"Error pushing results: {e}")


def _build_summary(evaluations):
    """Build a concise cross-architecture summary."""
    summary = {
        "experiment": "Cross-Architecture Validation",
        "description": "Same Progressive Cognitive method tested on Qwen 2.5 1.5B and Phi-2 2.7B",
        "models": {}
    }

    for model_name, results in evaluations.items():
        model_summary = {}
        for category, data in results.items():
            if isinstance(data, dict) and 'metrics' in data:
                metrics = data['metrics']
                model_summary[category] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                }
        summary["models"][model_name] = model_summary

    return summary


if __name__ == "__main__":
    main()
