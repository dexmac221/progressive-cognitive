"""
╔══════════════════════════════════════════════════════════════════╗
║   DREAM CYCLE EXPERIMENT — Continual Learning via SVD           ║
║                                                                  ║
║   Hypothesis: SVD compression (Dream Pruning) acts as a         ║
║   "sleep cycle" that consolidates knowledge and prevents         ║
║   catastrophic forgetting during continual learning.             ║
║                                                                  ║
║   Analogy: Sleep deprivation → hallucinations in humans          ║
║           No SVD cycles → hallucinations in LLMs                 ║
║                                                                  ║
║   Experiment Design:                                             ║
║     Model A (Dream Cycle):                                       ║
║       Dream-LoRA (arithmetic) → train logic → SVD → test         ║
║     Model B (Flat Continuous):                                   ║
║       Dream-LoRA (arithmetic) → train logic (no SVD) → test     ║
║     Model C (Fresh Logic):                                       ║
║       Fresh LoRA → train logic only → test                       ║
║                                                                  ║
║   Metrics:                                                       ║
║     - Arithmetic retention (does old knowledge survive?)         ║
║     - Logic acquisition (did it learn the new domain?)           ║
║     - Hallucination rate (does it make up confident garbage?)    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import random
import json
import time
import gc
from collections import defaultdict
from huggingface_hub import HfApi, snapshot_download


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

class DreamCycleConfig:
    model_name = "Qwen/Qwen2.5-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LoRA (must match original Dream-LoRA config)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # Logic training
    logic_samples = 3000
    logic_epochs = 3
    logic_lr = 1e-4
    batch_size = 4
    max_seq_len = 192     # logic tasks need more tokens

    # Dream Pruning (SVD)
    dream_pruning_rank = 8

    # Number of dream cycles (train → SVD → train → SVD)
    n_dream_cycles = 2

    # Source LoRA repos
    dream_lora_repo = "dexmac/progressive-cognitive-qwen3b-dream-lora"
    dream_lora_subfolder = "lora_adapters"


# ─────────────────────────────────────────────────────────────────
# LOGIC DOMAIN DATASET
# ─────────────────────────────────────────────────────────────────

class LogicDatasetGenerator:
    """
    Generates logic reasoning tasks — a completely different domain
    from the arithmetic the model was trained on.
    
    Task types:
    1. Comparison chains (A > B, B > C → A > C)
    2. Boolean logic (AND, OR, NOT)
    3. Set membership
    4. Syllogisms
    5. Pattern sequences
    6. Conditional reasoning (If...then)
    """

    NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    OBJECTS = ["apple", "book", "coin", "diamond", "egg", "flower", "gem", "hat"]
    COLORS = ["red", "blue", "green", "yellow", "white", "black", "purple", "orange"]

    @staticmethod
    def generate(n_samples):
        samples = []
        generators = [
            LogicDatasetGenerator._comparison_chain,
            LogicDatasetGenerator._boolean_logic,
            LogicDatasetGenerator._set_membership,
            LogicDatasetGenerator._syllogism,
            LogicDatasetGenerator._pattern_sequence,
            LogicDatasetGenerator._conditional_reasoning,
            LogicDatasetGenerator._negation_logic,
            LogicDatasetGenerator._transitivity,
        ]

        for _ in range(n_samples):
            gen = random.choice(generators)
            samples.append(gen())

        return samples

    @staticmethod
    def _comparison_chain():
        """A > B, B > C. Who is tallest?"""
        names = random.sample(LogicDatasetGenerator.NAMES, 3)
        attr = random.choice(["taller", "older", "faster", "heavier"])
        
        # Create chain: names[0] > names[1] > names[2]
        premise1 = f"{names[0]} is {attr} than {names[1]}"
        premise2 = f"{names[1]} is {attr} than {names[2]}"
        
        question_type = random.choice(["max", "min", "compare"])
        if question_type == "max":
            question = f"Who is the {attr.replace('er', 'est')}?"
            answer = names[0]
        elif question_type == "min":
            anti_attr = attr.replace("taller", "shortest").replace("older", "youngest").replace("faster", "slowest").replace("heavier", "lightest")
            question = f"Who is the {anti_attr}?"
            answer = names[2]
        else:
            question = f"Is {names[0]} {attr} than {names[2]}?"
            answer = "Yes"

        return (
            f"Logic: {premise1}. {premise2}. {question}\n"
            f"Answer: {answer}"
        )

    @staticmethod
    def _boolean_logic():
        """Evaluate boolean expressions."""
        a = random.choice([True, False])
        b = random.choice([True, False])
        
        op = random.choice(["AND", "OR"])
        negate = random.choice([True, False])
        
        if op == "AND":
            result = a and b
        else:
            result = a or b
        
        if negate:
            result = not result
            expr = f"NOT ({a} {op} {b})"
        else:
            expr = f"{a} {op} {b}"
        
        return (
            f"Logic: What is {expr}?\n"
            f"Answer: {result}"
        )

    @staticmethod
    def _set_membership():
        """Is X in the set {A, B, C}?"""
        items = random.sample(LogicDatasetGenerator.OBJECTS, 4)
        the_set = items[:3]
        
        query = random.choice([random.choice(the_set), items[3]])
        is_in = query in the_set
        
        set_str = ", ".join(the_set)
        return (
            f"Logic: The set contains: {{{set_str}}}. Is {query} in the set?\n"
            f"Answer: {'Yes' if is_in else 'No'}"
        )

    @staticmethod
    def _syllogism():
        """All A are B. X is A. Therefore X is B."""
        categories = [
            ("dogs", "animals", "mammals"),
            ("roses", "flowers", "plants"),
            ("cars", "vehicles", "machines"),
            ("triangles", "shapes", "polygons"),
            ("guitars", "instruments", "objects"),
        ]
        cat = random.choice(categories)
        
        scenario = random.choice(["valid", "invalid"])
        
        if scenario == "valid":
            instance = random.choice(["Rex", "Buddy", "Spot"]) if cat[0] == "dogs" else "this one"
            return (
                f"Logic: All {cat[0]} are {cat[1]}. {instance} is a {cat[0][:-1] if cat[0].endswith('s') else cat[0]}. "
                f"Is {instance} a {cat[1][:-1] if cat[1].endswith('s') else cat[1]}?\n"
                f"Answer: Yes"
            )
        else:
            return (
                f"Logic: All {cat[0]} are {cat[1]}. Something is a {cat[1][:-1] if cat[1].endswith('s') else cat[1]}. "
                f"Is it necessarily a {cat[0][:-1] if cat[0].endswith('s') else cat[0]}?\n"
                f"Answer: No"
            )

    @staticmethod
    def _pattern_sequence():
        """What comes next in the pattern?"""
        pattern_type = random.choice(["arithmetic", "repeat", "double"])
        
        if pattern_type == "arithmetic":
            start = random.randint(1, 20)
            step = random.randint(1, 10)
            seq = [start + step * i for i in range(5)]
            answer = seq[-1] + step
            seq_str = ", ".join(str(x) for x in seq)
            return (
                f"Logic: What comes next in the sequence: {seq_str}, ?\n"
                f"Answer: {answer}"
            )
        elif pattern_type == "repeat":
            colors = random.sample(LogicDatasetGenerator.COLORS, 3)
            pattern = colors * 2 + [colors[0]]
            pattern_str = ", ".join(pattern)
            answer = colors[1]
            return (
                f"Logic: What comes next in the pattern: {pattern_str}, ?\n"
                f"Answer: {answer}"
            )
        else:  # double
            start = random.randint(1, 5)
            seq = [start * (2 ** i) for i in range(5)]
            answer = seq[-1] * 2
            seq_str = ", ".join(str(x) for x in seq)
            return (
                f"Logic: What comes next in the sequence: {seq_str}, ?\n"
                f"Answer: {answer}"
            )

    @staticmethod
    def _conditional_reasoning():
        """If P then Q. P is true. What about Q?"""
        conditions = [
            ("it rains", "the ground is wet", "the ground is dry"),
            ("the light is on", "the room is bright", "the room is dark"),
            ("the door is locked", "nobody can enter", "anyone can enter"),
            ("the alarm rings", "everyone wakes up", "everyone sleeps"),
            ("the temperature drops below 0", "water freezes", "water stays liquid"),
        ]
        
        cond, consequent, opposite = random.choice(conditions)
        scenario = random.choice(["affirm", "deny", "inverse"])
        
        if scenario == "affirm":
            # Modus ponens: If P then Q. P. Therefore Q.
            return (
                f"Logic: If {cond}, then {consequent}. {cond.capitalize()}. "
                f"What follows?\n"
                f"Answer: {consequent.capitalize()}"
            )
        elif scenario == "deny":
            # Modus tollens: If P then Q. Not Q. Therefore not P.
            return (
                f"Logic: If {cond}, then {consequent}. {opposite.capitalize()}. "
                f"Does that mean {cond}?\n"
                f"Answer: No"
            )
        else:
            # Affirming the consequent (fallacy): If P then Q. Q. Therefore P? NO
            return (
                f"Logic: If {cond}, then {consequent}. {consequent.capitalize()}. "
                f"Can we conclude that {cond}?\n"
                f"Answer: No, that is a logical fallacy"
            )

    @staticmethod
    def _negation_logic():
        """Double negation, De Morgan's laws in natural language."""
        name = random.choice(LogicDatasetGenerator.NAMES)
        
        scenario = random.choice(["double_neg", "demorgan_and", "demorgan_or"])
        
        if scenario == "double_neg":
            statement = random.choice([
                (f"It is not true that {name} is not happy", f"{name} is happy"),
                (f"It is false that {name} did not attend", f"{name} attended"),
                (f"{name} is not unlike the others", f"{name} is like the others"),
            ])
            return (
                f"Logic: {statement[0]}. What does this mean?\n"
                f"Answer: {statement[1]}"
            )
        elif scenario == "demorgan_and":
            obj1, obj2 = random.sample(LogicDatasetGenerator.COLORS, 2)
            return (
                f"Logic: The ball is not ({obj1} and {obj2}). "
                f"Can the ball be {obj1}?\n"
                f"Answer: Yes, as long as it is not also {obj2}"
            )
        else:
            obj1, obj2 = random.sample(LogicDatasetGenerator.COLORS, 2)
            return (
                f"Logic: The ball is not ({obj1} or {obj2}). "
                f"Can the ball be {obj1}?\n"
                f"Answer: No"
            )

    @staticmethod
    def _transitivity():
        """If A=B and B=C then A=C."""
        names = random.sample(LogicDatasetGenerator.NAMES, 4)
        values = random.sample(range(10, 100), 4)
        
        scenario = random.choice(["equality", "inequality"])
        
        if scenario == "equality":
            return (
                f"Logic: {names[0]} has the same score as {names[1]}. "
                f"{names[1]} has the same score as {names[2]}. "
                f"{names[2]}'s score is {values[0]}. "
                f"What is {names[0]}'s score?\n"
                f"Answer: {values[0]}"
            )
        else:
            # A > B, B > C, C = value → A > value
            return (
                f"Logic: {names[0]} scored higher than {names[1]}. "
                f"{names[1]} scored higher than {names[2]}. "
                f"{names[2]} scored {values[0]}. "
                f"Is {names[0]}'s score greater than {values[0]}?\n"
                f"Answer: Yes"
            )


class LogicDataset(Dataset):
    """Dataset of logic reasoning tasks."""

    def __init__(self, tokenizer, n_samples, max_len=192):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = LogicDatasetGenerator.generate(n_samples)
        print(f"  Generated {len(self.data)} logic samples")
        print(f"  Example: '{self.data[0][:80]}...'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
        }


# ─────────────────────────────────────────────────────────────────
# DREAM PRUNER (SVD) — same as main training script
# ─────────────────────────────────────────────────────────────────

class DreamPruner:
    """SVD Low-Rank Factorization for LoRA weight compression."""

    @staticmethod
    def svd_prune(model, target_rank=8):
        stats = {'pruned_rank': target_rank, 'total_layers': 0}
        masks = {}

        lora_layers = defaultdict(dict)
        for name, param in model.named_parameters():
            if 'lora_A' in name and 'lora_A.default.weight' in name:
                base_name = name.replace('lora_A.default.weight', '')
                lora_layers[base_name]['A'] = (name, param)
            elif 'lora_B' in name and 'lora_B.default.weight' in name:
                base_name = name.replace('lora_B.default.weight', '')
                lora_layers[base_name]['B'] = (name, param)

        for base_name, parts in lora_layers.items():
            if 'A' not in parts or 'B' not in parts:
                continue

            name_A, param_A = parts['A']
            name_B, param_B = parts['B']

            A = param_A.data.float()
            B = param_B.data.float()

            r = A.shape[0]
            if r <= target_rank:
                continue

            Q_B, R_B = torch.linalg.qr(B)
            Q_A, R_A = torch.linalg.qr(A.T)

            C = R_B @ R_A.T
            U, S, Vh = torch.linalg.svd(C)

            U_k = U[:, :target_rank]
            S_k = S[:target_rank]
            Vh_k = Vh[:target_rank, :]

            B_new = Q_B @ U_k @ torch.diag(torch.sqrt(S_k))
            A_new = torch.diag(torch.sqrt(S_k)) @ Vh_k @ Q_A.T

            B_padded = torch.zeros_like(B)
            B_padded[:, :target_rank] = B_new

            A_padded = torch.zeros_like(A)
            A_padded[:target_rank, :] = A_new

            param_A.data.copy_(A_padded.to(param_A.dtype))
            param_B.data.copy_(B_padded.to(param_B.dtype))

            mask_A = torch.zeros_like(A)
            mask_A[:target_rank, :] = 1.0
            masks[name_A] = mask_A.to(param_A.device)

            mask_B = torch.zeros_like(B)
            mask_B[:, :target_rank] = 1.0
            masks[name_B] = mask_B.to(param_B.device)

            stats['total_layers'] += 1

        return stats, masks

    @staticmethod
    def apply_masks(model, masks):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data *= masks[name]


# ─────────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────

def train_on_logic(model, tokenizer, config, dataset, epochs, lr, masks=None):
    """Train model on logic dataset, optionally applying SVD masks."""
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Ensure model is in training mode BEFORE collecting trainable params
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"    Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    assert len(trainable_params) > 0, "No trainable parameters! Check LoRA loading."

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    history = []

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if masks:
                DreamPruner.apply_masks(model, masks)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return history


def load_dream_lora(config):
    """Load the pre-trained Dream-LoRA from HF Hub."""
    print("  Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
        trust_remote_code=True,
    )

    print(f"  Downloading Dream-LoRA from {config.dream_lora_repo}...")
    repo_path = snapshot_download(repo_id=config.dream_lora_repo)
    lora_path = os.path.join(repo_path, config.dream_lora_subfolder)

    print(f"  Applying LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
    # Ensure LoRA params are trainable for continual learning
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Dream-LoRA loaded. Trainable params: {trainable:,}")
    assert trainable > 0, "No trainable parameters found! LoRA loading failed."

    return model, tokenizer


def create_fresh_lora(config):
    """Create a fresh LoRA model (no pre-training) for control."""
    print("  Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)
    print(f"  Fresh LoRA created. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# QUICK ARITHMETIC TEST (for retention check during training)
# ─────────────────────────────────────────────────────────────────

def quick_arithmetic_test(model, tokenizer, config, n=20):
    """Quick check: can the model still do arithmetic?"""
    model.eval()
    correct = 0
    sensible = 0
    total = n

    random.seed(99)  # fixed seed for reproducibility
    for _ in range(n):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        op = random.choice(['+', '-', '*'])
        expr = f"{a} {op} {b}"
        exact = eval(expr)

        prompt = f"Calculate: {expr} ="
        inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract number
        import re
        nums = re.findall(r'-?\d+', response)
        predicted = int(nums[0]) if nums else None

        if predicted == exact:
            correct += 1
        elif predicted is not None:
            # Check if "sensible" (within 50% and same order of magnitude)
            if abs(exact) > 0:
                rel_error = abs(predicted - exact) / abs(exact)
                import math
                mag_diff = abs(
                    math.floor(math.log10(max(abs(predicted), 1))) -
                    math.floor(math.log10(max(abs(exact), 1)))
                )
                if rel_error < 0.5 and mag_diff <= 1:
                    sensible += 1

    random.seed()  # reset seed
    model.train()
    return {
        'exact_accuracy': correct / total * 100,
        'sensible_rate': (correct + sensible) / total * 100,
        'correct': correct,
        'sensible': sensible,
        'total': total,
    }


def quick_logic_test(model, tokenizer, config, n=20):
    """Quick check: can the model do logic?"""
    model.eval()
    correct = 0
    total = n

    random.seed(77)
    samples = LogicDatasetGenerator.generate(n)

    for sample in samples:
        # Split into prompt and expected answer
        parts = sample.split("\nAnswer: ")
        if len(parts) != 2:
            total -= 1
            continue
        prompt_text = parts[0] + "\nAnswer:"
        expected = parts[1].strip().lower()

        inputs = tokenizer(prompt_text, return_tensors="pt").to(config.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response_clean = response.strip().lower()

        # Flexible matching
        if expected in response_clean or response_clean.startswith(expected[:10]):
            correct += 1
        elif expected in ("yes", "no") and expected in response_clean.split()[:5]:
            correct += 1

    random.seed()
    model.train()
    return {
        'logic_accuracy': correct / max(total, 1) * 100,
        'correct': correct,
        'total': total,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────

def run_experiment():
    start_time = time.time()
    config = DreamCycleConfig()
    results = {}

    print("=" * 70)
    print("  DREAM CYCLE EXPERIMENT")
    print("  Continual Learning via SVD Compression Cycles")
    print("=" * 70)

    torch.manual_seed(42)
    random.seed(42)

    # ═══════════════════════════════════════════════════════════════
    # MODEL A: Dream Cycle (arithmetic LoRA → logic + SVD cycles)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL A: DREAM CYCLE")
    print("  Dream-LoRA (arithmetic) → logic training with SVD cycles")
    print("=" * 60)

    model_a, tokenizer = load_dream_lora(config)

    # Baseline arithmetic test BEFORE logic training
    print("\n  ── Pre-training arithmetic check ──")
    arith_before_a = quick_arithmetic_test(model_a, tokenizer, config)
    print(f"    Arithmetic accuracy: {arith_before_a['exact_accuracy']:.1f}%")
    print(f"    Sensible rate: {arith_before_a['sensible_rate']:.1f}%")

    # Dream Cycles: train logic → SVD → train logic → SVD
    masks = {}
    logic_history_a = []

    for cycle in range(config.n_dream_cycles):
        print(f"\n  ── Dream Cycle {cycle+1}/{config.n_dream_cycles} ──")

        # Train on logic
        print(f"  Training on logic domain...")
        dataset = LogicDataset(tokenizer, config.logic_samples // config.n_dream_cycles, config.max_seq_len)
        cycle_history = train_on_logic(
            model_a, tokenizer, config, dataset,
            config.logic_epochs, config.logic_lr, masks=masks
        )
        logic_history_a.extend(cycle_history)

        # SVD compression (the "dream")
        print(f"\n  SVD Dream Pruning (rank {config.lora_r} → {config.dream_pruning_rank})...")
        stats, masks = DreamPruner.svd_prune(model_a, config.dream_pruning_rank)
        print(f"    Compressed {stats['total_layers']} LoRA layers")

        # Quick check after cycle
        arith_check = quick_arithmetic_test(model_a, tokenizer, config, n=10)
        logic_check = quick_logic_test(model_a, tokenizer, config, n=10)
        print(f"    Post-cycle arithmetic: {arith_check['exact_accuracy']:.1f}%")
        print(f"    Post-cycle logic: {logic_check['logic_accuracy']:.1f}%")

    # Final tests
    print("\n  ── Final evaluation: Model A (Dream Cycle) ──")
    arith_final_a = quick_arithmetic_test(model_a, tokenizer, config, n=50)
    logic_final_a = quick_logic_test(model_a, tokenizer, config, n=50)
    print(f"    Arithmetic: {arith_final_a['exact_accuracy']:.1f}% exact, {arith_final_a['sensible_rate']:.1f}% sensible")
    print(f"    Logic: {logic_final_a['logic_accuracy']:.1f}%")

    results['model_a_dream_cycle'] = {
        'name': 'Dream Cycle (SVD continual learning)',
        'arithmetic_before': arith_before_a,
        'arithmetic_after': arith_final_a,
        'logic_after': logic_final_a,
        'training_history': logic_history_a,
        'n_cycles': config.n_dream_cycles,
    }

    # Save Model A
    out_a = "./output_dream_cycle/model_a"
    os.makedirs(out_a, exist_ok=True)
    model_a.save_pretrained(out_a)
    tokenizer.save_pretrained(out_a)

    # Free VRAM
    del model_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # MODEL B: Flat Continuous (arithmetic LoRA → logic, NO SVD)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL B: FLAT CONTINUOUS (Control)")
    print("  Dream-LoRA (arithmetic) → logic training WITHOUT SVD")
    print("=" * 60)

    torch.manual_seed(42)
    random.seed(42)

    model_b, tokenizer = load_dream_lora(config)

    # Baseline arithmetic test
    print("\n  ── Pre-training arithmetic check ──")
    arith_before_b = quick_arithmetic_test(model_b, tokenizer, config)
    print(f"    Arithmetic accuracy: {arith_before_b['exact_accuracy']:.1f}%")

    # Train on ALL logic data at once, no SVD
    print(f"\n  Training on logic domain (no SVD cycles)...")
    dataset = LogicDataset(tokenizer, config.logic_samples, config.max_seq_len)
    logic_history_b = train_on_logic(
        model_b, tokenizer, config, dataset,
        config.logic_epochs * config.n_dream_cycles,  # same total epochs
        config.logic_lr
    )

    # Final tests
    print("\n  ── Final evaluation: Model B (Flat Continuous) ──")
    arith_final_b = quick_arithmetic_test(model_b, tokenizer, config, n=50)
    logic_final_b = quick_logic_test(model_b, tokenizer, config, n=50)
    print(f"    Arithmetic: {arith_final_b['exact_accuracy']:.1f}% exact, {arith_final_b['sensible_rate']:.1f}% sensible")
    print(f"    Logic: {logic_final_b['logic_accuracy']:.1f}%")

    results['model_b_flat'] = {
        'name': 'Flat Continuous (no SVD, control)',
        'arithmetic_before': arith_before_b,
        'arithmetic_after': arith_final_b,
        'logic_after': logic_final_b,
        'training_history': logic_history_b,
    }

    # Save Model B
    out_b = "./output_dream_cycle/model_b"
    os.makedirs(out_b, exist_ok=True)
    model_b.save_pretrained(out_b)
    tokenizer.save_pretrained(out_b)

    del model_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # MODEL C: Fresh Logic (no arithmetic pre-training, control)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODEL C: FRESH LOGIC (Control)")
    print("  Fresh LoRA → logic training only (no arithmetic base)")
    print("=" * 60)

    torch.manual_seed(42)
    random.seed(42)

    model_c, tokenizer = create_fresh_lora(config)

    # Baseline arithmetic test (should be ~0%)
    print("\n  ── Pre-training arithmetic check ──")
    arith_before_c = quick_arithmetic_test(model_c, tokenizer, config)
    print(f"    Arithmetic accuracy: {arith_before_c['exact_accuracy']:.1f}% (expected ~base level)")

    # Train on logic
    print(f"\n  Training on logic domain...")
    dataset = LogicDataset(tokenizer, config.logic_samples, config.max_seq_len)
    logic_history_c = train_on_logic(
        model_c, tokenizer, config, dataset,
        config.logic_epochs * config.n_dream_cycles,
        config.logic_lr
    )

    # Final tests
    print("\n  ── Final evaluation: Model C (Fresh Logic) ──")
    arith_final_c = quick_arithmetic_test(model_c, tokenizer, config, n=50)
    logic_final_c = quick_logic_test(model_c, tokenizer, config, n=50)
    print(f"    Arithmetic: {arith_final_c['exact_accuracy']:.1f}% exact")
    print(f"    Logic: {logic_final_c['logic_accuracy']:.1f}%")

    results['model_c_fresh'] = {
        'name': 'Fresh Logic (no arithmetic, control)',
        'arithmetic_before': arith_before_c,
        'arithmetic_after': arith_final_c,
        'logic_after': logic_final_c,
        'training_history': logic_history_c,
    }

    # Save Model C
    out_c = "./output_dream_cycle/model_c"
    os.makedirs(out_c, exist_ok=True)
    model_c.save_pretrained(out_c)
    tokenizer.save_pretrained(out_c)

    del model_c
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # COMPARATIVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  DREAM CYCLE EXPERIMENT — RESULTS")
    print("=" * 70)

    print(f"\n  Time: {elapsed/60:.1f} minutes")
    print(f"\n  {'Model':<35s} | {'Arith Before':>13s} | {'Arith After':>12s} | {'Δ Arith':>8s} | {'Logic':>8s}")
    print(f"  {'─'*35} | {'─'*13} | {'─'*12} | {'─'*8} | {'─'*8}")

    for key in ['model_a_dream_cycle', 'model_b_flat', 'model_c_fresh']:
        r = results[key]
        before = r['arithmetic_before']['exact_accuracy']
        after = r['arithmetic_after']['exact_accuracy']
        delta = after - before
        logic = r['logic_after']['logic_accuracy']
        delta_str = f"{delta:+.1f}%"
        print(f"  {r['name']:<35s} | {before:>12.1f}% | {after:>11.1f}% | {delta_str:>8s} | {logic:>7.1f}%")

    # Key insight
    a_retention = results['model_a_dream_cycle']['arithmetic_after']['exact_accuracy']
    b_retention = results['model_b_flat']['arithmetic_after']['exact_accuracy']
    a_logic = results['model_a_dream_cycle']['logic_after']['logic_accuracy']
    b_logic = results['model_b_flat']['logic_after']['logic_accuracy']

    print(f"""
  ╔════════════════════════════════════════════════════════════════╗
  ║  KEY FINDINGS                                                  ║
  ╠════════════════════════════════════════════════════════════════╣
  ║  Arithmetic Retention:                                         ║
  ║    Dream Cycle: {a_retention:5.1f}%  vs  Flat Continuous: {b_retention:5.1f}%          ║
  ║  Logic Acquisition:                                            ║
  ║    Dream Cycle: {a_logic:5.1f}%  vs  Flat Continuous: {b_logic:5.1f}%          ║
  ║                                                                ║
  ║  If Dream > Flat on retention → SVD acts as "sleep"           ║
  ║  consolidating old knowledge while making room for new.       ║
  ╚════════════════════════════════════════════════════════════════╝
    """)

    # Save results
    results['config'] = {
        'model_name': config.model_name,
        'logic_samples': config.logic_samples,
        'logic_epochs': config.logic_epochs,
        'n_dream_cycles': config.n_dream_cycles,
        'dream_pruning_rank': config.dream_pruning_rank,
        'elapsed_minutes': elapsed / 60,
    }

    os.makedirs('./results', exist_ok=True)
    results_path = './results/dream_cycle_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Push to HF Hub
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-results")

    if hf_token:
        print(f"\n  Pushing results to {repo_id}...")
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=results_path,
                path_in_repo="dream_cycle_results.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            print("  Results pushed successfully!")
        except Exception as e:
            print(f"  Error pushing results: {e}")

    # Also push trained models
    if hf_token:
        for label, out_dir, repo_suffix in [
            ("Dream Cycle", "./output_dream_cycle/model_a", "dream-cycle-lora"),
            ("Flat Continuous", "./output_dream_cycle/model_b", "flat-continuous-lora"),
        ]:
            try:
                model_repo = f"dexmac/progressive-cognitive-{repo_suffix}"
                print(f"  Pushing {label} model to {model_repo}...")
                api.create_repo(repo_id=model_repo, exist_ok=True)
                api.upload_folder(
                    folder_path=out_dir,
                    repo_id=model_repo,
                    repo_type="model",
                    token=hf_token,
                )
                print(f"  {label} model pushed!")
            except Exception as e:
                print(f"  Error pushing {label}: {e}")

    # Pause Space
    space_id = os.environ.get("SPACE_ID")
    if space_id:
        try:
            api = HfApi(token=hf_token)
            print(f"\n  Pausing Space {space_id}...")
            api.pause_space(repo_id=space_id)
        except Exception as e:
            print(f"  Error pausing Space: {e}")


if __name__ == "__main__":
    run_experiment()
