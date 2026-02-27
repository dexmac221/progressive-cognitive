"""
╔══════════════════════════════════════════════════════════════════╗
║   ABLATION STUDY — Statistical Rigor & SVD Rank Analysis        ║
║                                                                  ║
║   1. 100 tests per category (vs 20 in pilot)                    ║
║   2. 3 runs with different seeds → mean ± std                   ║
║   3. SVD Rank ablation: rank 4, 8, 12                           ║
║                                                                  ║
║   Goal: make the results statistically robust and               ║
║   find the optimal compression sweet spot.                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json
import os
import numpy as np
from collections import defaultdict
from evaluation_framework import TestConfig, TestSuiteGenerator, ModelEvaluator, ComparativeReport


class RealModelEvaluator(ModelEvaluator):
    def __init__(self, name, config, model, tokenizer):
        super().__init__(name, config)
        self.model = model
        self.tokenizer = tokenizer
        
    def _simulate_response(self, prompt):
        """Override the simulation with real model inference."""
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


def load_model_from_hub(model_id, repo_id, subfolder=None, device="cuda:0"):
    """Load a base model + LoRA adapter from HF Hub."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    try:
        from huggingface_hub import snapshot_download
        print(f"  Downloading {repo_id}...")
        repo_path = snapshot_download(repo_id=repo_id)
        lora_path = os.path.join(repo_path, subfolder) if subfolder else repo_path
        
        if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            print(f"  Loading LoRA weights from {lora_path}...")
            model = PeftModel.from_pretrained(base_model, lora_path)
            model = model.merge_and_unload()
        else:
            print(f"  WARNING: adapter_config.json not found at {lora_path}")
            model = base_model
    except Exception as e:
        print(f"  Could not download {repo_id}: {e}")
        model = base_model
    
    model.eval()
    return model


def run_single_evaluation(models_dict, tokenizer, n_samples, seed, device):
    """Run one complete evaluation with a given seed and n_samples."""
    config = TestConfig(n_samples_per_test=n_samples, device=device, max_new_tokens=100, seed=seed)
    
    suite = TestSuiteGenerator.generate_all(n=config.n_samples_per_test, seed=config.seed)
    total_tests = sum(len(s['tests']) for s in suite.values())
    print(f"  Generated {total_tests} tests across {len(suite)} categories (seed={seed})")
    
    evaluations = {}
    for name, model in models_dict.items():
        print(f"  Evaluating {name}...")
        evaluator = RealModelEvaluator(name, config, model, tokenizer)
        evaluations[name] = evaluator.evaluate_suite(suite)
    
    return evaluations


def extract_metrics(evaluation):
    """Extract key metrics from an evaluation result."""
    metrics = {}
    for cat_name, cat_data in evaluation.items():
        s = cat_data['summary']
        metrics[cat_name] = {
            'accuracy': s.get('accuracy', 0),
            'sensible_error_rate': s.get('sensible_error_rate', None),
            'catastrophic_error_rate': s.get('catastrophic_error_rate', None),
            'delegation_accuracy': s.get('delegation_accuracy', None),
            'delegation_rate': s.get('delegation_rate', None),
        }
    return metrics


def compute_stats(runs_metrics):
    """Compute mean ± std across multiple runs."""
    stats = {}
    categories = runs_metrics[0].keys()
    
    for cat in categories:
        stats[cat] = {}
        metric_keys = runs_metrics[0][cat].keys()
        
        for key in metric_keys:
            values = [run[cat][key] for run in runs_metrics if run[cat][key] is not None]
            if values:
                stats[cat][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n_runs': len(values),
                    'values': values
                }
    return stats


def print_ablation_report(all_stats):
    """Print a comprehensive ablation report."""
    print("\n" + "═" * 80)
    print("  ABLATION STUDY — Statistical Analysis (3 runs × 100 tests)")
    print("  Mean ± Std across seeds [42, 123, 7]")
    print("═" * 80)
    
    models = list(all_stats.keys())
    categories = list(list(all_stats.values())[0].keys())
    
    # Header
    print(f"\n  {'METRIC':<28s}", end="")
    for model in models:
        short = model.replace('Qwen2.5-1.5B ', '').replace('+ ', '')[:16]
        print(f" │ {short:>22s}", end="")
    print()
    print(f"  {'─' * 28}", end="")
    for _ in models:
        print(f" │ {'─' * 22}", end="")
    print()
    
    for cat in categories:
        # Accuracy
        print(f"  {cat:<28s}", end="")
        for model in models:
            s = all_stats[model][cat].get('accuracy', {})
            if s:
                print(f" │ {s['mean']:>7.1f}% ± {s['std']:>4.1f}%", end="")
            else:
                print(f" │ {'N/A':>22s}", end="")
        print()
        
        # Sensible error rate
        if any(all_stats[m][cat].get('sensible_error_rate') for m in models):
            print(f"  {'  ↳ sensible errors':<28s}", end="")
            for model in models:
                s = all_stats[model][cat].get('sensible_error_rate', {})
                if s:
                    print(f" │ {s['mean']:>7.1f}% ± {s['std']:>4.1f}%", end="")
                else:
                    print(f" │ {'N/A':>22s}", end="")
            print()
        
        # Delegation accuracy
        if any(all_stats[m][cat].get('delegation_accuracy') for m in models):
            print(f"  {'  ↳ delegation acc.':<28s}", end="")
            for model in models:
                s = all_stats[model][cat].get('delegation_accuracy', {})
                if s:
                    print(f" │ {s['mean']:>7.1f}% ± {s['std']:>4.1f}%", end="")
                else:
                    print(f" │ {'N/A':>22s}", end="")
            print()
    
    print()


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   ABLATION STUDY — Progressive Cognitive Architecture           ║")
    print("║   100 tests × 3 seeds × 4 models = 6,000 evaluations           ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B"
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load all models once
    print("\n── Loading Models ──")
    
    print("\n[1/4] Loading Base Model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model_base.eval()
    
    print("\n[2/4] Loading Flat-LoRA...")
    model_flat = load_model_from_hub(model_id, "dexmac/progressive-cognitive-baseline-lora", device=device)
    
    print("\n[3/4] Loading Progressive-LoRA (Magnitude Pruning)...")
    model_prog = load_model_from_hub(model_id, "dexmac/progressive-cognitive-lora", subfolder="lora_adapters", device=device)
    
    print("\n[4/4] Loading Dream-LoRA (SVD Pruning, rank 8)...")
    model_dream = load_model_from_hub(model_id, "dexmac/progressive-cognitive-dream-lora", subfolder="lora_adapters", device=device)
    
    models_dict = {
        'Qwen2.5-1.5B (Base)': model_base,
        'Qwen2.5-1.5B + Flat LoRA': model_flat,
        'Qwen2.5-1.5B + Progressive LoRA': model_prog,
        'Qwen2.5-1.5B + Dream LoRA': model_dream,
    }
    
    # ═══════════════════════════════════════════════════════════
    # PART 1: Statistical rigor — 3 runs × 100 tests
    # ═══════════════════════════════════════════════════════════
    
    N_SAMPLES = 100
    SEEDS = [42, 123, 7]
    
    print(f"\n{'═' * 70}")
    print(f"  PART 1: Statistical Evaluation")
    print(f"  {N_SAMPLES} tests per category × {len(SEEDS)} seeds × {len(models_dict)} models")
    print(f"  Total evaluations: {N_SAMPLES * 5 * len(SEEDS) * len(models_dict)}")
    print(f"{'═' * 70}\n")
    
    all_runs = {name: [] for name in models_dict}
    all_raw_evaluations = []
    
    start_time = time.time()
    
    for i, seed in enumerate(SEEDS):
        print(f"\n── Run {i+1}/{len(SEEDS)} (seed={seed}) ──")
        evaluations = run_single_evaluation(models_dict, tokenizer, N_SAMPLES, seed, device)
        all_raw_evaluations.append(evaluations)
        
        for name in models_dict:
            metrics = extract_metrics(evaluations[name])
            all_runs[name].append(metrics)
    
    elapsed = time.time() - start_time
    print(f"\n  Total evaluation time: {elapsed/60:.1f} minutes")
    
    # Compute statistics
    all_stats = {}
    for name in models_dict:
        all_stats[name] = compute_stats(all_runs[name])
    
    # Print report
    print_ablation_report(all_stats)
    
    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY BOX
    # ═══════════════════════════════════════════════════════════
    
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  ABLATION STUDY CONCLUSIONS                                   ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    
    # Find best model per category
    categories = list(list(all_stats.values())[0].keys())
    for cat in categories:
        best_model = max(all_stats.keys(), key=lambda m: all_stats[m][cat].get('accuracy', {}).get('mean', 0))
        best_val = all_stats[best_model][cat]['accuracy']['mean']
        best_std = all_stats[best_model][cat]['accuracy']['std']
        short = best_model.replace('Qwen2.5-1.5B ', '').replace('+ ', '')
        print(f"  ║  {cat:<20s}: {short:<18s} ({best_val:.1f}% ± {best_std:.1f}%)  ║")
    
    print("  ╚════════════════════════════════════════════════════════════════╝")
    
    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    
    results = {
        'metadata': {
            'n_samples_per_test': N_SAMPLES,
            'seeds': SEEDS,
            'n_runs': len(SEEDS),
            'total_evaluations': N_SAMPLES * 5 * len(SEEDS) * len(models_dict),
            'elapsed_minutes': round(elapsed / 60, 1),
            'base_model': model_id,
        },
        'statistical_summary': {},
        'raw_per_run': []
    }
    
    # Statistical summary
    for name in models_dict:
        results['statistical_summary'][name] = all_stats[name]
    
    # Raw per-run data (the last seed's full evaluation for detailed error analysis)
    for i, seed in enumerate(SEEDS):
        run_data = {}
        for name in models_dict:
            run_data[name] = all_runs[name][i]
        results['raw_per_run'].append({
            'seed': seed,
            'metrics': run_data
        })
    
    os.makedirs('./results', exist_ok=True)
    report_path = './results/ablation_study.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {report_path}")
    
    # Push to Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-results")
    
    if hf_token:
        print(f"\n  Pushing results to Hugging Face Hub: {repo_id}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=report_path,
                path_in_repo="ablation_study.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token
            )
            print("  Successfully pushed ablation results to Hub!")
            
            # Pause the space to save money
            space_id = os.environ.get("SPACE_ID")
            if space_id:
                print(f"  Pausing space {space_id} to save resources...")
                api.pause_space(repo_id=space_id)
        except Exception as e:
            print(f"  Error pushing to hub: {e}")


if __name__ == "__main__":
    main()
