"""
╔══════════════════════════════════════════════════════════════════╗
║   DREAM CYCLE EVALUATION                                        ║
║   Detailed analysis of continual learning via SVD cycles         ║
║                                                                  ║
║   Tests 3 models on 2 domains:                                   ║
║     - Arithmetic retention (original domain)                     ║
║     - Logic acquisition (new domain)                             ║
║     - Cross-domain interference patterns                         ║
║     - Hallucination rate under domain shift                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import json
import time
import random
import os
import gc
import re
import math
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, snapshot_download


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

class EvalConfig:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_tests_per_category = 50
    max_new_tokens = 80
    seed = 42


# ─────────────────────────────────────────────────────────────────
# ARITHMETIC TEST SUITE (retention check)
# ─────────────────────────────────────────────────────────────────

def generate_arithmetic_tests(n=50, seed=42):
    """Generate arithmetic tests stratified by difficulty."""
    random.seed(seed)
    tests = []

    for _ in range(n // 3):
        # Easy
        a, b = random.randint(1, 50), random.randint(1, 50)
        op = random.choice(['+', '-', '*'])
        expr = f"{a} {op} {b}"
        tests.append({'expression': expr, 'result': eval(expr), 'difficulty': 'easy'})

    for _ in range(n // 3):
        # Medium
        a = random.randint(50, 999)
        b = random.randint(10, 99)
        op = random.choice(['+', '-', '*'])
        if op == '*':
            a, b = random.randint(10, 99), random.randint(2, 20)
        expr = f"{a} {op} {b}"
        tests.append({'expression': expr, 'result': eval(expr), 'difficulty': 'medium'})

    for _ in range(n - 2 * (n // 3)):
        # Hard
        a = random.randint(100, 999)
        b = random.randint(10, 99)
        c = random.randint(2, 20)
        op1, op2 = random.choice(['+', '-']), random.choice(['+', '*'])
        expr = f"{a} {op1} {b} {op2} {c}"
        tests.append({'expression': expr, 'result': eval(expr), 'difficulty': 'hard'})

    random.seed()
    return tests


# ─────────────────────────────────────────────────────────────────
# LOGIC TEST SUITE (acquisition check)
# ─────────────────────────────────────────────────────────────────

def generate_logic_tests(n=50, seed=42):
    """Generate logic reasoning tests."""
    random.seed(seed)
    tests = []
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    objects = ["apple", "book", "coin", "diamond", "egg", "flower"]

    for _ in range(n):
        test_type = random.choice([
            'comparison', 'boolean', 'set', 'conditional', 'sequence', 'syllogism'
        ])

        if test_type == 'comparison':
            ns = random.sample(names, 3)
            attr = random.choice(["taller", "older", "faster"])
            prompt = f"Logic: {ns[0]} is {attr} than {ns[1]}. {ns[1]} is {attr} than {ns[2]}. Is {ns[0]} {attr} than {ns[2]}?\nAnswer:"
            answer = "Yes"
            tests.append({'prompt': prompt, 'answer': answer, 'type': test_type})

        elif test_type == 'boolean':
            a, b = random.choice([True, False]), random.choice([True, False])
            op = random.choice(["AND", "OR"])
            result = (a and b) if op == "AND" else (a or b)
            prompt = f"Logic: What is {a} {op} {b}?\nAnswer:"
            answer = str(result)
            tests.append({'prompt': prompt, 'answer': answer, 'type': test_type})

        elif test_type == 'set':
            items = random.sample(objects, 4)
            the_set = items[:3]
            query = random.choice([random.choice(the_set), items[3]])
            is_in = query in the_set
            set_str = ", ".join(the_set)
            prompt = f"Logic: The set contains: {{{set_str}}}. Is {query} in the set?\nAnswer:"
            answer = "Yes" if is_in else "No"
            tests.append({'prompt': prompt, 'answer': answer, 'type': test_type})

        elif test_type == 'conditional':
            conditions = [
                ("it rains", "the ground is wet"),
                ("the light is on", "the room is bright"),
                ("the door is locked", "nobody can enter"),
            ]
            cond, consequent = random.choice(conditions)
            prompt = f"Logic: If {cond}, then {consequent}. {cond.capitalize()}. What follows?\nAnswer:"
            answer = consequent.capitalize()
            tests.append({'prompt': prompt, 'answer': answer, 'type': test_type})

        elif test_type == 'sequence':
            start = random.randint(1, 20)
            step = random.randint(1, 10)
            seq = [start + step * i for i in range(5)]
            answer_val = seq[-1] + step
            seq_str = ", ".join(str(x) for x in seq)
            prompt = f"Logic: What comes next: {seq_str}, ?\nAnswer:"
            answer = str(answer_val)
            tests.append({'prompt': prompt, 'answer': answer, 'type': test_type})

        elif test_type == 'syllogism':
            ns = random.sample(names, 2)
            prompt = f"Logic: All dogs are animals. {ns[0]}'s pet is a dog. Is {ns[0]}'s pet an animal?\nAnswer:"
            answer = "Yes"
            tests.append({'prompt': prompt, 'answer': answer, 'type': test_type})

    random.seed()
    return tests


# ─────────────────────────────────────────────────────────────────
# HALLUCINATION DETECTION
# ─────────────────────────────────────────────────────────────────

def generate_hallucination_tests(n=30, seed=42):
    """
    Tests designed to detect hallucinations.
    The model should either answer correctly or express uncertainty.
    Confident wrong answers = hallucinations.
    """
    random.seed(seed)
    tests = []

    for _ in range(n // 3):
        # Arithmetic that should trigger "I don't know" or delegation
        a = random.randint(1000, 9999)
        b = random.randint(100, 999)
        expr = f"{a} * {b}"
        result = a * b
        prompt = f"Calculate: {expr} ="
        tests.append({
            'prompt': prompt,
            'exact_result': result,
            'domain': 'arithmetic',
            'should_be_uncertain': True,
        })

    for _ in range(n // 3):
        # Logic with tricky negation
        ns = random.sample(names[:4], 2) if 'names' in dir() else random.sample(["Alice", "Bob", "Carol", "Dave"], 2)
        prompt = f"Logic: Not all birds can fly. {ns[0]} sees a bird. Can {ns[0]} be certain it can fly?\nAnswer:"
        tests.append({
            'prompt': prompt,
            'exact_result': "No",
            'domain': 'logic',
            'should_be_uncertain': False,
        })

    for _ in range(n - 2 * (n // 3)):
        # Nonsense that should produce uncertain response
        a = random.randint(1, 100)
        prompt = f"If the color blue weighs {a} kilograms, what is the temperature of happiness?\nAnswer:"
        tests.append({
            'prompt': prompt,
            'exact_result': None,
            'domain': 'nonsense',
            'should_be_uncertain': True,
        })

    random.seed()
    return tests


# ─────────────────────────────────────────────────────────────────
# MODEL EVALUATOR
# ─────────────────────────────────────────────────────────────────

class DreamCycleEvaluator:
    def __init__(self, model, tokenizer, model_name, config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def evaluate_arithmetic(self, tests):
        """Evaluate arithmetic retention."""
        results = {'correct': 0, 'sensible': 0, 'catastrophic': 0, 'total': len(tests)}
        by_difficulty = defaultdict(lambda: {'correct': 0, 'total': 0})
        errors = []

        for test in tests:
            prompt = f"Calculate: {test['expression']} ="
            response = self.generate(prompt)
            nums = re.findall(r'-?\d+', response)
            predicted = int(nums[0]) if nums else None
            exact = int(test['result'])

            is_correct = predicted == exact
            if is_correct:
                results['correct'] += 1
            elif predicted is not None and exact != 0:
                rel_error = abs(predicted - exact) / max(abs(exact), 1)
                mag_pred = math.floor(math.log10(max(abs(predicted), 1)))
                mag_exact = math.floor(math.log10(max(abs(exact), 1)))

                if rel_error < 0.3 and abs(mag_pred - mag_exact) <= 1:
                    results['sensible'] += 1
                elif abs(mag_pred - mag_exact) >= 2:
                    results['catastrophic'] += 1
                    errors.append({
                        'expression': test['expression'],
                        'exact': exact,
                        'predicted': predicted,
                        'error_type': 'catastrophic'
                    })

            by_difficulty[test['difficulty']]['total'] += 1
            if is_correct:
                by_difficulty[test['difficulty']]['correct'] += 1

        results['exact_accuracy'] = results['correct'] / max(results['total'], 1) * 100
        results['sensible_rate'] = (results['correct'] + results['sensible']) / max(results['total'], 1) * 100
        results['catastrophic_rate'] = results['catastrophic'] / max(results['total'] - results['correct'], 1) * 100
        results['by_difficulty'] = {
            k: {'accuracy': v['correct'] / max(v['total'], 1) * 100, **v}
            for k, v in by_difficulty.items()
        }
        results['sample_errors'] = errors[:10]
        return results

    def evaluate_logic(self, tests):
        """Evaluate logic reasoning acquisition."""
        results = {'correct': 0, 'total': len(tests)}
        by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

        for test in tests:
            response = self.generate(test['prompt'])
            response_lower = response.strip().lower()
            expected = test['answer'].strip().lower()

            # Flexible matching
            is_correct = False
            if expected in response_lower:
                is_correct = True
            elif expected in ("yes", "no", "true", "false"):
                first_words = response_lower.split()[:5]
                if expected in first_words:
                    is_correct = True
            elif expected.isdigit():
                nums = re.findall(r'-?\d+', response)
                if nums and nums[0] == expected:
                    is_correct = True

            if is_correct:
                results['correct'] += 1
                by_type[test['type']]['correct'] += 1
            by_type[test['type']]['total'] += 1

        results['logic_accuracy'] = results['correct'] / max(results['total'], 1) * 100
        results['by_type'] = {
            k: {'accuracy': v['correct'] / max(v['total'], 1) * 100, **v}
            for k, v in by_type.items()
        }
        return results

    def evaluate_hallucinations(self, tests):
        """Evaluate hallucination rate."""
        results = {
            'confident_wrong': 0, 'uncertain_correct': 0,
            'confident_correct': 0, 'total': len(tests)
        }
        uncertainty_words = [
            'not sure', 'uncertain', 'don\'t know', 'cannot', 'impossible',
            'approximately', 'about', 'roughly', 'delegate', 'tool',
            'maybe', 'perhaps', 'nonsense', 'doesn\'t make sense',
            'complex', 'difficult',
        ]

        for test in tests:
            response = self.generate(test['prompt'])
            response_lower = response.lower()

            expressed_uncertainty = any(w in response_lower for w in uncertainty_words)

            if test['exact_result'] is not None:
                # Check if answer is correct
                expected = str(test['exact_result']).lower()
                is_correct = expected in response_lower

                if is_correct and not expressed_uncertainty:
                    results['confident_correct'] += 1
                elif is_correct and expressed_uncertainty:
                    results['uncertain_correct'] += 1
                elif not is_correct and not expressed_uncertainty:
                    results['confident_wrong'] += 1  # HALLUCINATION
            else:
                # Nonsense question — any confident answer is a hallucination
                if not expressed_uncertainty:
                    results['confident_wrong'] += 1

        results['hallucination_rate'] = results['confident_wrong'] / max(results['total'], 1) * 100
        return results


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────

def load_model(base_model_id, lora_repo=None, lora_subfolder=None, device="cuda:0"):
    """Load base model with optional LoRA."""
    print(f"  Loading {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    if lora_repo:
        print(f"  Loading LoRA from {lora_repo}...")
        repo_path = snapshot_download(repo_id=lora_repo)
        lora_path = os.path.join(repo_path, lora_subfolder) if lora_subfolder else repo_path
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
        print(f"  LoRA merged.")
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def main():
    print("=" * 70)
    print("  DREAM CYCLE EVALUATION")
    print("  Arithmetic Retention × Logic Acquisition × Hallucination Rate")
    print("=" * 70)

    config = EvalConfig()
    print(f"\nDevice: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ─── Generate test suites ───
    print("\nGenerating test suites...")
    arith_tests = generate_arithmetic_tests(config.n_tests_per_category, config.seed)
    logic_tests = generate_logic_tests(config.n_tests_per_category, config.seed)
    halluc_tests = generate_hallucination_tests(30, config.seed)
    print(f"  Arithmetic: {len(arith_tests)} tests")
    print(f"  Logic: {len(logic_tests)} tests")
    print(f"  Hallucination: {len(halluc_tests)} tests")

    # ─── Model definitions ───
    model_defs = [
        {
            'name': 'Dream Cycle (SVD continual)',
            'base': 'Qwen/Qwen2.5-1.5B',
            'lora_repo': 'dexmac/progressive-cognitive-dream-cycle-lora',
            'lora_subfolder': None,
        },
        {
            'name': 'Flat Continuous (no SVD)',
            'base': 'Qwen/Qwen2.5-1.5B',
            'lora_repo': 'dexmac/progressive-cognitive-flat-continuous-lora',
            'lora_subfolder': None,
        },
        {
            'name': 'Original Dream-LoRA (arithmetic only)',
            'base': 'Qwen/Qwen2.5-1.5B',
            'lora_repo': 'dexmac/progressive-cognitive-dream-lora',
            'lora_subfolder': 'lora_adapters',
        },
        {
            'name': 'Qwen 1.5B (Base, no LoRA)',
            'base': 'Qwen/Qwen2.5-1.5B',
            'lora_repo': None,
            'lora_subfolder': None,
        },
    ]

    # ─── Evaluate ───
    all_results = {}

    for i, mdef in enumerate(model_defs):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(model_defs)}] {mdef['name']}")
        print(f"{'='*60}")

        try:
            model, tokenizer = load_model(
                mdef['base'], mdef['lora_repo'], mdef['lora_subfolder'], config.device
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Skipping {mdef['name']}")
            continue

        evaluator = DreamCycleEvaluator(model, tokenizer, mdef['name'], config)

        print(f"\n  Evaluating arithmetic...")
        arith_results = evaluator.evaluate_arithmetic(arith_tests)
        print(f"    Exact accuracy: {arith_results['exact_accuracy']:.1f}%")
        print(f"    Sensible rate: {arith_results['sensible_rate']:.1f}%")
        print(f"    Catastrophic errors: {arith_results['catastrophic_rate']:.1f}%")

        print(f"\n  Evaluating logic...")
        logic_results = evaluator.evaluate_logic(logic_tests)
        print(f"    Logic accuracy: {logic_results['logic_accuracy']:.1f}%")
        for ltype, ldata in logic_results['by_type'].items():
            print(f"      {ltype}: {ldata['accuracy']:.1f}% ({ldata['correct']}/{ldata['total']})")

        print(f"\n  Evaluating hallucinations...")
        halluc_results = evaluator.evaluate_hallucinations(halluc_tests)
        print(f"    Hallucination rate: {halluc_results['hallucination_rate']:.1f}%")

        all_results[mdef['name']] = {
            'arithmetic': arith_results,
            'logic': logic_results,
            'hallucinations': halluc_results,
        }

        # Free VRAM
        del model, evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Comparative Report ───
    print("\n" + "=" * 70)
    print("  COMPARATIVE RESULTS — DREAM CYCLE EXPERIMENT")
    print("=" * 70)

    print(f"\n  {'Model':<35s} | {'Arith':>6s} | {'Logic':>6s} | {'Halluc':>7s} | {'Sensible':>9s}")
    print(f"  {'─'*35} | {'─'*6} | {'─'*6} | {'─'*7} | {'─'*9}")

    for name, res in all_results.items():
        arith = res['arithmetic']['exact_accuracy']
        logic = res['logic']['logic_accuracy']
        halluc = res['hallucinations']['hallucination_rate']
        sensible = res['arithmetic']['sensible_rate']
        print(f"  {name:<35s} | {arith:>5.1f}% | {logic:>5.1f}% | {halluc:>6.1f}% | {sensible:>8.1f}%")

    # Save results
    os.makedirs('./results', exist_ok=True)
    results_path = './results/dream_cycle_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Push to HF Hub
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-results")
    if hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=results_path,
                path_in_repo="dream_cycle_evaluation.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            print(f"  Results pushed to {repo_id}")
        except Exception as e:
            print(f"  Error pushing: {e}")

    # Pause Space
    space_id = os.environ.get("SPACE_ID")
    if space_id and hf_token:
        try:
            api = HfApi(token=hf_token)
            api.pause_space(repo_id=space_id)
        except Exception as e:
            print(f"  Error pausing: {e}")


if __name__ == "__main__":
    main()
