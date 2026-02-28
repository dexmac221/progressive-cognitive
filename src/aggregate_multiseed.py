"""
╔══════════════════════════════════════════════════════════════════╗
║   AGGREGATE MULTI-SEED RESULTS                                   ║
║                                                                  ║
║   Downloads seed 42, 43, 44 summaries from HF Hub and computes  ║
║   mean ± std for each metric × model pair.                       ║
║   Outputs publication-ready tables.                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import os
import numpy as np
from collections import defaultdict
from huggingface_hub import hf_hub_download

TOKEN = os.environ.get('HF_TOKEN')
assert TOKEN, 'Set HF_TOKEN environment variable'
DATASET = 'dexmac/progressive-cognitive-results'
SEEDS = [42, 43, 44]


def download_summaries():
    """Download all seed summary files from HF Hub."""
    summaries = {}
    for seed in SEEDS:
        fname = f'english_eval_seed{seed}_summary.json'
        try:
            path = hf_hub_download(
                repo_id=DATASET,
                filename=fname,
                repo_type='dataset',
                token=TOKEN,
            )
            with open(path) as f:
                summaries[seed] = json.load(f)
            print(f"  ✅ Downloaded seed {seed}")
        except Exception as e:
            print(f"  ❌ Seed {seed}: {e}")
    return summaries


def extract_metrics(summaries):
    """
    Extract key metrics per model per seed.
    
    Returns: {model_name: {metric_name: [val_seed42, val_seed43, val_seed44]}}
    Values are in percentage (0-100).
    """
    metrics = defaultdict(lambda: defaultdict(list))
    
    for seed in SEEDS:
        if seed not in summaries:
            print(f"  ⚠ Missing seed {seed}, skipping")
            continue
        
        data = summaries[seed]
        
        for model_name, categories in data.items():
            m = metrics[model_name]
            
            # Exact accuracy
            if 'exact_accuracy' in categories:
                ea = categories['exact_accuracy']
                m['exact_accuracy'].append(ea.get('accuracy', 0))
                m['sensible_error_rate_ea'].append(ea.get('sensible_error_rate', 0))
                m['catastrophic_error_rate_ea'].append(ea.get('catastrophic_error_rate', 0))
            
            # Number sense (strict + magnitude)
            if 'number_sense' in categories:
                ns = categories['number_sense']
                m['number_sense_strict'].append(ns.get('number_sense_rate', 0))
                m['magnitude_sense'].append(ns.get('magnitude_sense_rate', 0))
            
            # Self-awareness / Delegation
            if 'self_awareness' in categories:
                sa = categories['self_awareness']
                m['self_aware_rate'].append(sa.get('accuracy', 0))
                m['delegation_accuracy'].append(sa.get('delegation_accuracy', 0))
                m['delegation_rate'].append(sa.get('delegation_rate', 0))
            
            # Adversarial
            if 'adversarial' in categories:
                adv = categories['adversarial']
                m['adversarial_accuracy'].append(adv.get('accuracy', 0))
            
            # Error patterns
            if 'error_patterns' in categories:
                ep = categories['error_patterns']
                m['sensible_error_rate'].append(ep.get('sensible_error_rate', 0))
                m['catastrophic_error_rate'].append(ep.get('catastrophic_error_rate', 0))
    
    return metrics


def format_mean_std(values):
    """Format as 'XX.X ± Y.Y%'. Values are already in percentage (0-100)."""
    if not values:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{mean:.1f} ± {std:.1f}%"


def format_table(metrics):
    """Print publication-ready comparison table."""
    
    # Short names for display
    short_names = {
        'Qwen2.5-1.5B (Base)': '1.5B Base',
        'Qwen2.5-1.5B + Flat LoRA (EN)': '1.5B Flat',
        'Qwen2.5-1.5B + Dream (EN)': '1.5B Dream',
        'Qwen2.5-3B (Base)': '3B Base',
        'Qwen2.5-3B + Flat LoRA (EN)': '3B Flat',
        'Qwen2.5-3B + Dream (EN)': '3B Dream',
    }
    
    display_metrics = [
        ('exact_accuracy', 'Exact Accuracy'),
        ('number_sense_strict', 'Number Sense (strict)'),
        ('magnitude_sense', 'Magnitude Sense (OoM±1)'),
        ('self_aware_rate', 'Self-Aware Rate'),
        ('delegation_accuracy', 'Delegation Accuracy'),
        ('delegation_rate', 'Delegation Rate'),
        ('adversarial_accuracy', 'Adversarial Robustness'),
        ('sensible_error_rate', 'Sensible Errors'),
        ('catastrophic_error_rate', 'Catastrophic Errors'),
    ]
    
    # Ordered models
    model_order = [
        'Qwen2.5-1.5B (Base)',
        'Qwen2.5-1.5B + Flat LoRA (EN)',
        'Qwen2.5-1.5B + Dream (EN)',
        'Qwen2.5-3B (Base)',
        'Qwen2.5-3B + Flat LoRA (EN)',
        'Qwen2.5-3B + Dream (EN)',
    ]
    
    print("\n" + "="*120)
    print("  PUBLICATION-READY RESULTS: Mean ± Std across 3 seeds (42, 43, 44)")
    print("="*120)
    
    # Header
    header = f"{'Metric':<28}"
    for model in model_order:
        header += f" | {short_names.get(model, model):>18}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for metric_key, metric_label in display_metrics:
        row = f"{metric_label:<28}"
        for model in model_order:
            vals = metrics.get(model, {}).get(metric_key, [])
            row += f" | {format_mean_std(vals):>18}"
        print(row)
    
    print("=" * 120)
    
    # Also output LaTeX table
    print("\n\n% LaTeX Table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Progressive Cognitive Architecture: English Evaluation Results (mean $\\pm$ std, $n=3$ seeds)}")
    print("\\label{tab:results}")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l" + "c" * len(model_order) + "}")
    print("\\toprule")
    
    latex_header = "\\textbf{Metric}"
    for model in model_order:
        latex_header += f" & \\textbf{{{short_names.get(model, model)}}}"
    print(latex_header + " \\\\")
    print("\\midrule")
    
    for metric_key, metric_label in display_metrics:
        row = metric_label.replace('%', '\\%')
        for model in model_order:
            vals = metrics.get(model, {}).get(metric_key, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                cell = f"${mean:.1f} \\pm {std:.1f}$"
            else:
                cell = "N/A"
            row += f" & {cell}"
        print(row + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{table}")


def save_aggregate(metrics, filename='aggregate_results.json'):
    """Save aggregate results as JSON."""
    output = {}
    for model, mets in metrics.items():
        output[model] = {}
        for metric_key, vals in mets.items():
            if vals:
                output[model][metric_key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    'values': [float(v) for v in vals],
                    'n_seeds': len(vals),
                }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✅ Saved aggregate results to {filename}")


def main():
    print("Downloading seed summaries...")
    summaries = download_summaries()
    
    if len(summaries) < 2:
        print(f"\n  ⚠ Only {len(summaries)} seeds available. Need at least 2 for std.")
        print("  Re-run when more results are available.")
        return
    
    print(f"\nProcessing {len(summaries)} seeds: {list(summaries.keys())}")
    metrics = extract_metrics(summaries)
    
    format_table(metrics)
    save_aggregate(metrics, 'aggregate_results.json')
    
    # Quick highlight
    print("\n" + "="*80)
    print("  KEY FINDINGS")
    print("="*80)
    
    dream_15 = metrics.get('Qwen2.5-1.5B + Dream (EN)', {})
    flat_15 = metrics.get('Qwen2.5-1.5B + Flat LoRA (EN)', {})
    base_15 = metrics.get('Qwen2.5-1.5B (Base)', {})
    
    for metric_key, label in [
        ('exact_accuracy', 'Exact Accuracy'),
        ('adversarial_accuracy', 'Adversarial Robustness'),
        ('catastrophic_error_rate', 'Catastrophic Errors'),
        ('magnitude_sense', 'Magnitude Sense (OoM±1)'),
    ]:
        dream_vals = dream_15.get(metric_key, [])
        flat_vals = flat_15.get(metric_key, [])
        base_vals = base_15.get(metric_key, [])
        
        if dream_vals and flat_vals and base_vals:
            print(f"  {label}:")
            print(f"    Base: {format_mean_std(base_vals)}")
            print(f"    Flat: {format_mean_std(flat_vals)}")
            print(f"    Dream: {format_mean_std(dream_vals)}")


if __name__ == '__main__':
    main()
