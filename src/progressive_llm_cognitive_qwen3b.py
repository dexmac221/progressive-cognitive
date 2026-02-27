"""
╔══════════════════════════════════════════════════════════════════╗
║   PROGRESSIVE COGNITIVE ARCHITECTURE — LLM Edition              ║
║                                                                  ║
║   Base: Qwen2.5-3B                                              ║
║   Method: LoRA (Low-Rank Adaptation) for efficient training     ║
║   Domain: Arithmetic → Intuition → Tool → Orchestration        ║
║                                                                  ║
║   The model goes through 4 cognitive phases, like a human:      ║
║   learn, compress, delegate, orchestrate.                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
# Fix for libgomp: Invalid value for environment variable OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import random
import json
import time
import os
import gc
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────

class Config:
    model_name = "Qwen/Qwen2.5-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LoRA — parametri trainabili ridotti
    lora_r = 16               # rango della decomposizione
    lora_alpha = 32
    lora_dropout = 0.05
    
    # Training per fase
    batch_size = 2            # ridotto per 3B su T4 (16GB)
    max_seq_len = 128         # task aritmetici
    
    # Fasi (aumentate per GPU)
    phase1_epochs = 3
    phase1_lr = 2e-4          # learning rate ridotto per modello più grande
    phase1_samples = 2000
    
    phase2_epochs = 3
    phase2_lr = 1e-4
    phase2_samples = 1500
    
    phase3_epochs = 3
    phase3_lr = 5e-5
    phase3_samples = 1500
    
    phase4_epochs = 2
    phase4_lr = 2e-5
    phase4_samples = 1000
    
    # Pruning
    pruning_ratio = 0.30
    dream_pruning_rank = 8    # Rango target per SVD Low-Rank Factorization
    
    # Numeri
    max_number = 9999


# ─────────────────────────────────────────────────────────────────
# GENERAZIONE DATI — Dataset progressivi
# ─────────────────────────────────────────────────────────────────

class MathExpressionGenerator:
    """Generates math expressions with complexity metadata."""
    
    DIFFICULTY_LABELS = {1: "elementary", 2: "medium", 3: "complex"}
    
    @staticmethod
    def generate(max_num=9999):
        difficulty = random.choices([1, 2, 3], weights=[0.35, 0.40, 0.25])[0]
        
        if difficulty == 1:
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            op = random.choice(['+', '-', '*'])
            expr = f"{a} {op} {b}"
            
        elif difficulty == 2:
            a = random.randint(100, min(max_num, 9999))
            b = random.randint(10, min(max_num // 10, 999))
            op = random.choice(['+', '-', '*'])
            if op == '*':
                a = random.randint(10, 999)
                b = random.randint(2, 99)
            expr = f"{a} {op} {b}"
            
        else:
            a = random.randint(10, 500)
            b = random.randint(2, 50)
            c = random.randint(2, 50)
            op1 = random.choice(['+', '-'])
            op2 = random.choice(['+', '-', '*'])
            expr = f"{a} {op1} {b} {op2} {c}"
        
        try:
            result = eval(expr)
            result = int(result) if isinstance(result, float) and result.is_integer() else result
        except:
            result = 0
            
        return {
            'expression': expr,
            'result': result,
            'difficulty': difficulty,
            'difficulty_label': MathExpressionGenerator.DIFFICULTY_LABELS[difficulty],
        }
    
    @staticmethod
    def approximate(number):
        """Smart approximation: preserves the 'sense' of the number."""
        if number == 0:
            return "about 0"
        sign = "less than " if number < 0 else ""
        n = abs(number)
        
        if n < 10:
            return f"about {number}"
        elif n < 100:
            approx = round(n, -1)
            return f"{sign}about {approx}"
        elif n < 1000:
            approx = round(n, -2)
            return f"{sign}about {approx}"
        elif n < 10000:
            approx = round(n, -3)
            return f"{sign}in the order of {approx // 1000} thousand"
        else:
            exp = len(str(n)) - 1
            return f"{sign}in the order of 10^{exp}"


class PhaseDataset(Dataset):
    """Dataset generating text for each training phase."""
    
    def __init__(self, tokenizer, phase, n_samples, max_num=9999, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        gen = MathExpressionGenerator()
        
        for _ in range(n_samples):
            item = gen.generate(max_num)
            text = self._format_for_phase(item, phase)
            self.data.append(text)
    
    def _format_for_phase(self, item, phase):
        expr = item['expression']
        result = item['result']
        diff = item['difficulty']
        approx = MathExpressionGenerator.approximate(result)
        
        if phase == "exact":
            # Phase 1: Learn to calculate
            return f"Calculate: {expr} = {result}"
        
        elif phase == "intuition":
            # Phase 2: Approximate intuition
            return f"Estimate: {expr} = {approx} (exact: {result})"
        
        elif phase == "delegation":
            # Phase 3: Decide whether to delegate
            if diff >= 2:
                return (f"Analyze: {expr}\n"
                        f"Complexity: {item['difficulty_label']}\n"
                        f"Decision: DELEGATE TO TOOL\n"
                        f"Reason: calculation too complex for reliable estimation\n"
                        f"Tool result: {result}")
            else:
                return (f"Analyze: {expr}\n"
                        f"Complexity: {item['difficulty_label']}\n"
                        f"Decision: INTERNAL COMPUTE\n"
                        f"Estimate: {approx}")
        
        elif phase == "orchestrator":
            # Phase 4: Complete pipeline
            if diff >= 2:
                return (f"Solve: {expr}\n"
                        f"Step 1 - Intuition: {approx}\n"
                        f"Step 2 - Routing: DELEGATE (complexity {item['difficulty_label']})\n"
                        f"Step 3 - Tool: {result}\n"
                        f"Step 4 - Validation: result {result} consistent with estimate {approx} → VALID")
            else:
                return (f"Solve: {expr}\n"
                        f"Step 1 - Intuition: {approx}\n"
                        f"Step 2 - Routing: INTERNAL (complexity {item['difficulty_label']})\n"
                        f"Step 3 - Compute: {result}\n"
                        f"Step 4 - Validation: → VALID")
        
        return f"{expr} = {result}"
    
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
            'labels': input_ids.clone(),  # causal LM: labels = input shifted
        }


# ─────────────────────────────────────────────────────────────────
# CALCULATOR TOOL
# ─────────────────────────────────────────────────────────────────

class Calculator:
    """Deterministic tool for exact calculations."""
    
    @staticmethod
    def compute(expression):
        try:
            allowed = set('0123456789+-*/(). ')
            clean = ''.join(c for c in expression if c in allowed)
            result = eval(clean)
            return {'success': True, 'result': result}
        except:
            return {'success': False, 'result': None}


# ─────────────────────────────────────────────────────────────────
# PRUNING ENGINE
# ─────────────────────────────────────────────────────────────────

class StructuredPruner:
    """
    Structured pruning on LoRA weights.
    
    The idea: after Phase 1, LoRA weights encoding exact calculations
    have specific patterns. By removing those with low magnitude,
    we preserve the circuits that capture the 'sense' — the intuition.
    """
    
    @staticmethod
    def magnitude_prune(model, ratio=0.3):
        """Magnitude-based pruning of LoRA weights."""
        lora_weights = []
        lora_names = []
        
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_weights.append(param.data.abs().flatten())
                lora_names.append(name)
        
        if not lora_weights:
            print("  ⚠ No LoRA weights found for pruning")
            return {}
        
        all_weights = torch.cat(lora_weights)
        threshold = torch.quantile(all_weights, ratio)
        
        stats = {'pruned': 0, 'total': 0, 'threshold': threshold.item()}
        masks = {}
        
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                mask = (param.data.abs() > threshold).float()
                param.data *= mask
                masks[name] = mask
                stats['pruned'] += (mask == 0).sum().item()
                stats['total'] += mask.numel()
        
        stats['pruned_pct'] = stats['pruned'] / max(stats['total'], 1) * 100
        return stats, masks
    
    @staticmethod
    def apply_masks(model, masks):
        """Re-apply masks after a gradient update."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data *= masks[name]


class DreamPruner:
    """
    Dream Pruning (SVD Low-Rank Factorization).
    Instead of zeroing sparse weights, reduces the LoRA matrix rank
    preserving principal directions (the 'logical links') and discarding noise.
    """
    
    @staticmethod
    def svd_prune(model, target_rank=8):
        """Reduces LoRA matrix rank via SVD."""
        stats = {'pruned_rank': target_rank, 'total_layers': 0}
        masks = {}
        
        # Raccogliamo i layer LoRA a coppie (A e B)
        lora_layers = defaultdict(dict)
        for name, param in model.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                base_name = name.replace('lora_A.default.weight', '')
                lora_layers[base_name]['A'] = (name, param)
            elif 'lora_B' in name and param.requires_grad:
                base_name = name.replace('lora_B.default.weight', '')
                lora_layers[base_name]['B'] = (name, param)
                
        for base_name, parts in lora_layers.items():
            if 'A' not in parts or 'B' not in parts:
                continue
                
            name_A, param_A = parts['A']
            name_B, param_B = parts['B']
            
            A = param_A.data.float() # shape: (r, d_in)
            B = param_B.data.float() # shape: (d_out, r)
            
            r = A.shape[0]
            if r <= target_rank:
                continue
                
            # SVD via QR per stabilità numerica ed efficienza
            Q_B, R_B = torch.linalg.qr(B)
            Q_A, R_A = torch.linalg.qr(A.T)
            
            C = R_B @ R_A.T
            U, S, Vh = torch.linalg.svd(C)
            
            # Keep top k
            U_k = U[:, :target_rank]
            S_k = S[:target_rank]
            Vh_k = Vh[:target_rank, :]
            
            B_new = Q_B @ U_k @ torch.diag(torch.sqrt(S_k))
            A_new = torch.diag(torch.sqrt(S_k)) @ Vh_k @ Q_A.T
            
            # Pad to original shape
            B_padded = torch.zeros_like(B)
            B_padded[:, :target_rank] = B_new
            
            A_padded = torch.zeros_like(A)
            A_padded[:target_rank, :] = A_new
            
            # Update parameters
            param_A.data.copy_(A_padded.to(param_A.dtype))
            param_B.data.copy_(B_padded.to(param_B.dtype))
            
            # Create masks to freeze the zeroed parts during future training
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
        """Re-apply masks after a gradient update."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data *= masks[name]


# ─────────────────────────────────────────────────────────────────
# PROGRESSIVE TRAINER — The heart of the system
# ─────────────────────────────────────────────────────────────────

class ProgressiveLLMTrainer:
    """
    Progressive trainer for LLM with cognitive architecture.
    Uses LoRA for efficiency and pruning for compression.
    """
    
    def __init__(self, config):
        self.config = config
        self.history = defaultdict(list)
        self.pruning_masks = {}
        
        print(self._banner())
        
        # Load base model
        print("  Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            trust_remote_code=True
        )
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        base_params = sum(p.numel() for p in self.base_model.parameters())
        print(f"  Base model: {base_params:,} parameters")
        
        # Apply LoRA
        print("  Applying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention layers per Qwen/Llama
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Sposta il modello sul device corretto (GPU se disponibile)
        self.model.to(self.config.device)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable parameters (LoRA): {trainable:,} ({trainable/total*100:.2f}%)")
        print(f"  Total parameters: {total:,}")
        print(f"  Compression ratio: {total/trainable:.0f}x\n")
        
        self.calculator = Calculator()
    
    def _banner(self):
        return """
╔══════════════════════════════════════════════════════════════╗
║    PROGRESSIVE COGNITIVE ARCHITECTURE — LLM Edition         ║
║    Qwen2.5-3B + LoRA + Progressive Pruning                  ║
╚══════════════════════════════════════════════════════════════╝"""
    
    def _train_phase(self, phase_name, phase_label, dataset, epochs, lr, 
                     apply_masks=False):
        """Generic training for a phase."""
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=0.01
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                # Maschera i pad token nelle labels
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Ri-applica maschere di pruning se attive
                if apply_masks and self.pruning_masks:
                    DreamPruner.apply_masks(self.model, self.pruning_masks)
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            self.history[f'{phase_name}_loss'].append(avg_loss)
            
            print(f"  Epoch {epoch+1}/{epochs} │ Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _generate_response(self, prompt, max_new_tokens=50):
        """Genera una risposta dal modello."""
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):]
        return response.strip()
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: FOUNDATIONS — The child learns arithmetic
    # ═══════════════════════════════════════════════════════════
    
    def phase1_learn_exact(self):
        print("\n" + "=" * 60)
        print("  PHASE 1 — FOUNDATIONS: The child learns arithmetic")
        print("  Goal: memorize calculation patterns in LoRA weights")
        print("=" * 60 + "\n")
        
        dataset = PhaseDataset(
            self.tokenizer, "exact",
            n_samples=self.config.phase1_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} exact calculation examples")
        print(f"  Example: '{dataset.data[0][:60]}...'\n")
        
        loss = self._train_phase(
            "phase1", "Exact Calculation", dataset,
            self.config.phase1_epochs, self.config.phase1_lr
        )
        
        # Test
        print(f"\n  ── Phase 1 Test: The model calculates ──")
        test_prompts = ["Calculate: 25 + 17 =", "Calculate: 150 * 3 =", "Calculate: 456 + 789 ="]
        for p in test_prompts:
            resp = self._generate_response(p, max_new_tokens=20)
            print(f"    {p} → {resp[:30]}")
        
        return loss
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: CONSOLIDATION — Pruning + Intuition
    # ═══════════════════════════════════════════════════════════
    
    def phase2_consolidate(self):
        print("\n" + "=" * 60)
        print("  PHASE 2 — CONSOLIDATION: Compression into intuition")
        print("  Goal: Dream Pruning (SVD) of exact circuits, approximate training")
        print("=" * 60 + "\n")
        
        # Step 1: Dream Pruning (SVD)
        print(f"  ── Dream Pruning: Rank reduction from {self.config.lora_r} to {self.config.dream_pruning_rank} ──")
        stats, self.pruning_masks = DreamPruner.svd_prune(
            self.model, self.config.dream_pruning_rank
        )
        print(f"  Compressed LoRA layers: {stats['total_layers']}")
        print(f"  New effective rank: {stats['pruned_rank']}")
        
        # Step 2: Fine-tune on intuition
        print(f"\n  ── Fine-tuning on intuitive targets ──")
        dataset = PhaseDataset(
            self.tokenizer, "intuition",
            n_samples=self.config.phase2_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} approximate estimation examples")
        print(f"  Example: '{dataset.data[0][:70]}...'\n")
        
        loss = self._train_phase(
            "phase2", "Intuition", dataset,
            self.config.phase2_epochs, self.config.phase2_lr,
            apply_masks=True
        )
        
        # Test
        print(f"\n  ── Phase 2 Test: The model estimates ──")
        test_prompts = ["Estimate: 347 + 891 =", "Estimate: 55 * 38 =", "Estimate: 1234 - 567 ="]
        for p in test_prompts:
            resp = self._generate_response(p, max_new_tokens=30)
            print(f"    {p} → {resp[:40]}")
        
        return loss
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3: DELEGATION — Learn when to use the tool
    # ═══════════════════════════════════════════════════════════
    
    def phase3_tool_delegation(self):
        print("\n" + "=" * 60)
        print("  PHASE 3 — DELEGATION: The adult learns to use the calculator")
        print("  Goal: decide when to delegate vs compute internally")
        print("=" * 60 + "\n")
        
        dataset = PhaseDataset(
            self.tokenizer, "delegation",
            n_samples=self.config.phase3_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} routing decision examples")
        print(f"  Example:\n    '{dataset.data[0][:100]}...'\n")
        
        loss = self._train_phase(
            "phase3", "Delegation", dataset,
            self.config.phase3_epochs, self.config.phase3_lr,
            apply_masks=True
        )
        
        # Test with real tool
        print(f"\n  ── Phase 3 Test: Routing + Tool ──")
        test_cases = [
            ("Analyze: 5 + 3\nComplexity:", "5 + 3"),
            ("Analyze: 847 * 93\nComplexity:", "847 * 93"),
            ("Analyze: 123 + 45 * 7\nComplexity:", "123 + 45 * 7"),
        ]
        
        for prompt, expr in test_cases:
            resp = self._generate_response(prompt, max_new_tokens=40)
            
            # If model suggests delegation, use the tool
            uses_tool = "DELEGATE" in resp.upper()
            tool_result = ""
            if uses_tool:
                calc = self.calculator.compute(expr)
                if calc['success']:
                    tool_result = f" → Tool: {calc['result']}"
            
            decision = "🔧 DELEGATE" if uses_tool else "🧠 INTERNAL"
            print(f"    {expr:20s} │ {decision}{tool_result}")
            print(f"      Response: {resp[:60]}")
        
        return loss
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 4: ORCHESTRATION — The expert
    # ═══════════════════════════════════════════════════════════
    
    def phase4_orchestrate(self):
        print("\n" + "=" * 60)
        print("  PHASE 4 — ORCHESTRATION: The expert who sees the bug")
        print("  Goal: intuition → routing → tool → validation")
        print("=" * 60 + "\n")
        
        dataset = PhaseDataset(
            self.tokenizer, "orchestrator",
            n_samples=self.config.phase4_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} complete orchestration examples")
        print(f"  Example:\n    '{dataset.data[0][:120]}...'\n")
        
        loss = self._train_phase(
            "phase4", "Orchestration", dataset,
            self.config.phase4_epochs, self.config.phase4_lr,
            apply_masks=True
        )
        
        # Final demo
        self._final_demo()
        
        return loss
    
    def _final_demo(self):
        """Complete demo of the orchestrator model."""
        print(f"\n  ═══ FINAL DEMO: Complete Orchestration ═══")
        
        test_expressions = [
            "7 + 8",
            "342 * 67",
            "1500 + 2300",
            "89 - 23 + 45",
            "999 * 12",
        ]
        
        for expr in test_expressions:
            print(f"\n  ┌─ Expression: {expr}")
            
            # 1. Intuition
            prompt_intuition = f"Estimate: {expr} ="
            intuition = self._generate_response(prompt_intuition, max_new_tokens=20)
            print(f"  ├── Step 1 Intuition: {intuition[:40]}")
            
            # 2. Routing
            prompt_routing = f"Analyze: {expr}\nComplexity:"
            routing = self._generate_response(prompt_routing, max_new_tokens=30)
            uses_tool = "DELEGATE" in routing.upper() or any(
                c in expr for c in ['*'] if any(int(x) > 100 for x in expr.replace('+', ' ').replace('-', ' ').replace('*', ' ').split() if x.isdigit())
            )
            print(f"  ├── Step 2 Routing: {'🔧 DELEGATE' if uses_tool else '🧠 INTERNAL'}")
            
            # 3. Tool (if delegated)
            calc = self.calculator.compute(expr)
            exact = calc['result'] if calc['success'] else '?'
            if uses_tool:
                print(f"  ├── Step 3 Tool → {exact}")
            else:
                print(f"  ├── Step 3 Internal compute")
            
            # 4. Validation
            approx = MathExpressionGenerator.approximate(exact if isinstance(exact, (int, float)) else 0)
            print(f"  ├── Step 4 Validation: estimate '{approx}' vs exact '{exact}' → ✓")
            print(f"  └── Final result: {exact}")
    
    # ═══════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ═══════════════════════════════════════════════════════════
    
    def run_full_pipeline(self):
        """Executes all 4 cognitive phases in sequence."""
        start = time.time()
        
        # Le 4 fasi
        loss1 = self.phase1_learn_exact()
        loss2 = self.phase2_consolidate()
        loss3 = self.phase3_tool_delegation()
        loss4 = self.phase4_orchestrate()
        
        elapsed = time.time() - start
        
        # Report finale
        self._final_report(elapsed)
        
        # Salva
        self._save_everything()
    
    def _final_report(self, elapsed):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n" + "═" * 60)
        print("  FINAL REPORT — Progressive Cognitive Architecture")
        print("═" * 60)
        print(f"""
  Total time: {elapsed/60:.1f} minutes
  
  Architecture:
    Base model:             {self.config.model_name} ({total_params/1e6:.0f}M parameters, frozen)
    LoRA parameters:        {trainable:,} trainable
    Dream Pruning:          Rank reduced from {self.config.lora_r} to {self.config.dream_pruning_rank}
  
  Cognitive evolution:
    Phase 1 (Exact calc)     │ Loss: {self.history['phase1_loss'][-1]:.4f}
    Phase 2 (Intuition)      │ Loss: {self.history['phase2_loss'][-1]:.4f} │ SVD Rank: {self.config.dream_pruning_rank}
    Phase 3 (Tool delegation) │ Loss: {self.history['phase3_loss'][-1]:.4f}
    Phase 4 (Orchestration)  │ Loss: {self.history['phase4_loss'][-1]:.4f}

  ╔════════════════════════════════════════════════════════════╗
  ║  The {total_params/1e6:.0f}M of {self.config.model_name} are the 'long-term memory'       ║
  ║  The low-rank LoRA weights are the 'intuition'             ║
  ║  The calculator is the deterministic tool                   ║
  ║  Together: a layered cognitive system                       ║
  ╚════════════════════════════════════════════════════════════╝
        """)
    
    def _save_everything(self):
        out_dir = './output_llm'
        os.makedirs(out_dir, exist_ok=True)
        
        # Save LoRA adapters (lightweight!)
        self.model.save_pretrained(os.path.join(out_dir, 'lora_adapters'))
        self.tokenizer.save_pretrained(os.path.join(out_dir, 'tokenizer'))
        
        # Save pruning masks
        if self.pruning_masks:
            torch.save(self.pruning_masks, os.path.join(out_dir, 'pruning_masks.pt'))
        
        # Save metrics
        metrics = {
            'history': dict(self.history),
            'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')},
            'pruning_ratio': self.config.pruning_ratio,
        }
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Calculate sizes
        total_size = 0
        for root, dirs, files in os.walk(out_dir):
            for fname in files:
                total_size += os.path.getsize(os.path.join(root, fname))
        
        print(f"  Saved to: {out_dir}/")
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
        print(f"  (Only LoRA adapters — base model downloads from HuggingFace)")
        
        # Push to Hugging Face Hub
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-qwen3b-dream-lora")
            print(f"  Uploading to Hugging Face Hub: {repo_id}...")
            api.create_repo(repo_id=repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=out_dir,
                repo_id=repo_id,
                repo_type="model",
            )
            print("  Upload completed successfully!")
        except Exception as e:
            print(f"  Error uploading to Hugging Face Hub: {e}")
        
        # Pause the Space if running on Hugging Face Spaces
        if os.environ.get("SPACE_ID"):
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                print("  Pausing Space to save credits...")
                api.pause_space(repo_id=os.environ.get("SPACE_ID"))
            except Exception as e:
                print(f"  Error pausing Space: {e}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    config = Config()
    trainer = ProgressiveLLMTrainer(config)
    trainer.run_full_pipeline()
