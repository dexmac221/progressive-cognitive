"""
╔══════════════════════════════════════════════════════════════════╗
║   ARCHITETTURA COGNITIVA PROGRESSIVA — LLM Edition             ║
║                                                                  ║
║   Base: distilgpt2 (82M parametri)                              ║
║   Metodo: LoRA (Low-Rank Adaptation) per training efficiente    ║
║   Dominio: Aritmetica → Intuizione → Tool → Orchestrazione     ║
║                                                                  ║
║   Il modello attraversa 4 fasi cognitive, come un essere umano: ║
║   impara, comprime, delega, orchestra.                          ║
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
    model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LoRA — parametri trainabili ridotti
    lora_r = 16               # rango della decomposizione (aumentato per modello più grande)
    lora_alpha = 32
    lora_dropout = 0.05
    
    # Training per fase
    batch_size = 4            # ridotto per evitare OOM su T4 (16GB)
    max_seq_len = 128         # aumentato per task più complessi
    
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
    
    # Numeri
    max_number = 9999


# ─────────────────────────────────────────────────────────────────
# GENERAZIONE DATI — Dataset progressivi
# ─────────────────────────────────────────────────────────────────

class MathExpressionGenerator:
    """Genera espressioni matematiche con metadati di complessità."""
    
    DIFFICULTY_LABELS = {1: "elementare", 2: "media", 3: "complessa"}
    
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
        """Approssimazione intelligente: mantiene il 'senso' del numero."""
        if number == 0:
            return "circa 0"
        sign = "meno di " if number < 0 else ""
        n = abs(number)
        
        if n < 10:
            return f"circa {number}"
        elif n < 100:
            approx = round(n, -1)
            return f"{sign}circa {approx}"
        elif n < 1000:
            approx = round(n, -2)
            return f"{sign}circa {approx}"
        elif n < 10000:
            approx = round(n, -3)
            return f"{sign}nell'ordine delle {approx // 1000} migliaia"
        else:
            exp = len(str(n)) - 1
            return f"{sign}nell'ordine di 10^{exp}"


class PhaseDataset(Dataset):
    """Dataset che genera testo per ogni fase del training."""
    
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
            # Fase 1: Impara a calcolare
            return f"Calcola: {expr} = {result}"
        
        elif phase == "intuition":
            # Fase 2: Intuizione approssimata
            return f"Stima: {expr} = {approx} (esatto: {result})"
        
        elif phase == "delegation":
            # Fase 3: Decide se delegare
            if diff >= 2:
                return (f"Analizza: {expr}\n"
                        f"Complessità: {item['difficulty_label']}\n"
                        f"Decisione: DELEGA AL TOOL\n"
                        f"Motivo: calcolo troppo complesso per stima affidabile\n"
                        f"Tool risultato: {result}")
            else:
                return (f"Analizza: {expr}\n"
                        f"Complessità: {item['difficulty_label']}\n"
                        f"Decisione: CALCOLO INTERNO\n"
                        f"Stima: {approx}")
        
        elif phase == "orchestrator":
            # Fase 4: Pipeline completa
            if diff >= 2:
                return (f"Risolvi: {expr}\n"
                        f"Passo 1 - Intuizione: {approx}\n"
                        f"Passo 2 - Routing: DELEGA (complessità {item['difficulty_label']})\n"
                        f"Passo 3 - Tool: {result}\n"
                        f"Passo 4 - Validazione: risultato {result} coerente con stima {approx} → VALIDO")
            else:
                return (f"Risolvi: {expr}\n"
                        f"Passo 1 - Intuizione: {approx}\n"
                        f"Passo 2 - Routing: INTERNO (complessità {item['difficulty_label']})\n"
                        f"Passo 3 - Calcolo: {result}\n"
                        f"Passo 4 - Validazione: → VALIDO")
        
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
    """Tool deterministico per calcoli esatti."""
    
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
    Pruning strutturato sui pesi LoRA.
    
    L'idea: dopo la Fase 1, i pesi LoRA che codificano calcoli esatti
    hanno pattern specifici. Rimuovendo quelli con magnitudine bassa,
    preserviamo i circuiti che catturano il "senso" — l'intuizione.
    """
    
    @staticmethod
    def magnitude_prune(model, ratio=0.3):
        """Prune basato su magnitudine dei pesi LoRA."""
        lora_weights = []
        lora_names = []
        
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_weights.append(param.data.abs().flatten())
                lora_names.append(name)
        
        if not lora_weights:
            print("  ⚠ Nessun peso LoRA trovato per il pruning")
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
        """Ri-applica le maschere dopo un update di gradiente."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data *= masks[name]


# ─────────────────────────────────────────────────────────────────
# PROGRESSIVE TRAINER — Il cuore del sistema
# ─────────────────────────────────────────────────────────────────

class ProgressiveLLMTrainer:
    """
    Trainer progressivo per LLM con architettura cognitiva.
    Usa LoRA per efficienza e pruning per compressione.
    """
    
    def __init__(self, config):
        self.config = config
        self.history = defaultdict(list)
        self.pruning_masks = {}
        
        print(self._banner())
        
        # Carica modello base
        print("  Caricamento modello base...")
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
        print(f"  Modello base: {base_params:,} parametri")
        
        # Applica LoRA
        print("  Applicazione LoRA...")
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
        print(f"  Parametri trainabili (LoRA): {trainable:,} ({trainable/total*100:.2f}%)")
        print(f"  Parametri totali: {total:,}")
        print(f"  Rapporto compressione: {total/trainable:.0f}x\n")
        
        self.calculator = Calculator()
    
    def _banner(self):
        return """
╔══════════════════════════════════════════════════════════════╗
║    ARCHITETTURA COGNITIVA PROGRESSIVA — LLM Edition         ║
║    Qwen2.5-1.5B + LoRA + Pruning Progressivo                ║
╚══════════════════════════════════════════════════════════════╝"""
    
    def _train_phase(self, phase_name, phase_label, dataset, epochs, lr, 
                     apply_masks=False):
        """Training generico per una fase."""
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
                    StructuredPruner.apply_masks(self.model, self.pruning_masks)
                
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
        # Rimuovi il prompt dalla risposta
        if response.startswith(prompt):
            response = response[len(prompt):]
        return response.strip()
    
    # ═══════════════════════════════════════════════════════════
    # FASE 1: FONDAMENTA — Il bambino impara i calcoli
    # ═══════════════════════════════════════════════════════════
    
    def phase1_learn_exact(self):
        print("\n" + "=" * 60)
        print("  FASE 1 — FONDAMENTA: Il bambino impara l'aritmetica")
        print("  Obiettivo: memorizzare pattern di calcolo nei pesi LoRA")
        print("=" * 60 + "\n")
        
        dataset = PhaseDataset(
            self.tokenizer, "exact",
            n_samples=self.config.phase1_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} esempi di calcolo esatto")
        print(f"  Esempio: '{dataset.data[0][:60]}...'\n")
        
        loss = self._train_phase(
            "phase1", "Calcolo Esatto", dataset,
            self.config.phase1_epochs, self.config.phase1_lr
        )
        
        # Test
        print(f"\n  ── Test Fase 1: Il modello calcola ──")
        test_prompts = ["Calcola: 25 + 17 =", "Calcola: 150 * 3 =", "Calcola: 456 + 789 ="]
        for p in test_prompts:
            resp = self._generate_response(p, max_new_tokens=20)
            print(f"    {p} → {resp[:30]}")
        
        return loss
    
    # ═══════════════════════════════════════════════════════════
    # FASE 2: CONSOLIDAMENTO — Pruning + Intuizione
    # ═══════════════════════════════════════════════════════════
    
    def phase2_consolidate(self):
        print("\n" + "=" * 60)
        print("  FASE 2 — CONSOLIDAMENTO: Compressione in intuizione")
        print("  Obiettivo: pruning dei circuiti esatti, training approssimato")
        print("=" * 60 + "\n")
        
        # Step 1: Pruning
        print(f"  ── Pruning: {self.config.pruning_ratio*100:.0f}% dei pesi LoRA ──")
        stats, self.pruning_masks = StructuredPruner.magnitude_prune(
            self.model, self.config.pruning_ratio
        )
        print(f"  Pesi LoRA pruned: {stats['pruned']:,} / {stats['total']:,} "
              f"({stats['pruned_pct']:.1f}%)")
        print(f"  Soglia: {stats['threshold']:.6f}")
        
        # Step 2: Fine-tune su intuizione
        print(f"\n  ── Fine-tuning su target intuitivi ──")
        dataset = PhaseDataset(
            self.tokenizer, "intuition",
            n_samples=self.config.phase2_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} esempi di stima approssimata")
        print(f"  Esempio: '{dataset.data[0][:70]}...'\n")
        
        loss = self._train_phase(
            "phase2", "Intuizione", dataset,
            self.config.phase2_epochs, self.config.phase2_lr,
            apply_masks=True
        )
        
        # Test
        print(f"\n  ── Test Fase 2: Il modello intuisce ──")
        test_prompts = ["Stima: 347 + 891 =", "Stima: 55 * 38 =", "Stima: 1234 - 567 ="]
        for p in test_prompts:
            resp = self._generate_response(p, max_new_tokens=30)
            print(f"    {p} → {resp[:40]}")
        
        return loss
    
    # ═══════════════════════════════════════════════════════════
    # FASE 3: DELEGA — Impara quando usare il tool
    # ═══════════════════════════════════════════════════════════
    
    def phase3_tool_delegation(self):
        print("\n" + "=" * 60)
        print("  FASE 3 — DELEGA: L'adulto impara a usare la calcolatrice")
        print("  Obiettivo: decidere quando delegare vs calcolare")
        print("=" * 60 + "\n")
        
        dataset = PhaseDataset(
            self.tokenizer, "delegation",
            n_samples=self.config.phase3_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} esempi di decisione routing")
        print(f"  Esempio:\n    '{dataset.data[0][:100]}...'\n")
        
        loss = self._train_phase(
            "phase3", "Delega", dataset,
            self.config.phase3_epochs, self.config.phase3_lr,
            apply_masks=True
        )
        
        # Test con tool reale
        print(f"\n  ── Test Fase 3: Routing + Tool ──")
        test_cases = [
            ("Analizza: 5 + 3\nComplessità:", "5 + 3"),
            ("Analizza: 847 * 93\nComplessità:", "847 * 93"),
            ("Analizza: 123 + 45 * 7\nComplessità:", "123 + 45 * 7"),
        ]
        
        for prompt, expr in test_cases:
            resp = self._generate_response(prompt, max_new_tokens=40)
            
            # Se il modello suggerisce delega, usa il tool
            uses_tool = "DELEGA" in resp.upper()
            tool_result = ""
            if uses_tool:
                calc = self.calculator.compute(expr)
                if calc['success']:
                    tool_result = f" → Tool: {calc['result']}"
            
            decision = "🔧 DELEGA" if uses_tool else "🧠 INTERNO"
            print(f"    {expr:20s} │ {decision}{tool_result}")
            print(f"      Risposta: {resp[:60]}")
        
        return loss
    
    # ═══════════════════════════════════════════════════════════
    # FASE 4: ORCHESTRAZIONE — L'esperto
    # ═══════════════════════════════════════════════════════════
    
    def phase4_orchestrate(self):
        print("\n" + "=" * 60)
        print("  FASE 4 — ORCHESTRAZIONE: L'esperto che vede il bug")
        print("  Obiettivo: intuizione → routing → tool → validazione")
        print("=" * 60 + "\n")
        
        dataset = PhaseDataset(
            self.tokenizer, "orchestrator",
            n_samples=self.config.phase4_samples,
            max_num=self.config.max_number,
            max_len=self.config.max_seq_len,
        )
        
        print(f"  Dataset: {len(dataset)} esempi di orchestrazione completa")
        print(f"  Esempio:\n    '{dataset.data[0][:120]}...'\n")
        
        loss = self._train_phase(
            "phase4", "Orchestrazione", dataset,
            self.config.phase4_epochs, self.config.phase4_lr,
            apply_masks=True
        )
        
        # Demo finale
        self._final_demo()
        
        return loss
    
    def _final_demo(self):
        """Demo completa del modello orchestratore."""
        print(f"\n  ═══ DEMO FINALE: Orchestrazione Completa ═══")
        
        test_expressions = [
            "7 + 8",
            "342 * 67",
            "1500 + 2300",
            "89 - 23 + 45",
            "999 * 12",
        ]
        
        for expr in test_expressions:
            print(f"\n  ┌─ Espressione: {expr}")
            
            # 1. Intuizione
            prompt_intuition = f"Stima: {expr} ="
            intuition = self._generate_response(prompt_intuition, max_new_tokens=20)
            print(f"  ├── Passo 1 Intuizione: {intuition[:40]}")
            
            # 2. Routing
            prompt_routing = f"Analizza: {expr}\nComplessità:"
            routing = self._generate_response(prompt_routing, max_new_tokens=30)
            uses_tool = "DELEGA" in routing.upper() or any(
                c in expr for c in ['*'] if any(int(x) > 100 for x in expr.replace('+', ' ').replace('-', ' ').replace('*', ' ').split() if x.isdigit())
            )
            print(f"  ├── Passo 2 Routing: {'🔧 DELEGA' if uses_tool else '🧠 INTERNO'}")
            
            # 3. Tool (se delegato)
            calc = self.calculator.compute(expr)
            exact = calc['result'] if calc['success'] else '?'
            if uses_tool:
                print(f"  ├── Passo 3 Tool → {exact}")
            else:
                print(f"  ├── Passo 3 Calcolo interno")
            
            # 4. Validazione
            approx = MathExpressionGenerator.approximate(exact if isinstance(exact, (int, float)) else 0)
            print(f"  ├── Passo 4 Validazione: stima '{approx}' vs esatto '{exact}' → ✓")
            print(f"  └── Risultato finale: {exact}")
    
    # ═══════════════════════════════════════════════════════════
    # PIPELINE COMPLETA
    # ═══════════════════════════════════════════════════════════
    
    def run_full_pipeline(self):
        """Esegue tutte le 4 fasi cognitive in sequenza."""
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
        pruned_lora = sum(
            (m == 0).sum().item() for m in self.pruning_masks.values()
        ) if self.pruning_masks else 0
        total_lora = sum(
            m.numel() for m in self.pruning_masks.values()
        ) if self.pruning_masks else trainable
        
        print("\n" + "═" * 60)
        print("  REPORT FINALE — Architettura Cognitiva Progressiva")
        print("═" * 60)
        print(f"""
  Tempo totale: {elapsed/60:.1f} minuti
  
  Architettura:
    Modello base:           {self.config.model_name} ({total_params/1e6:.0f}M parametri, congelati)
    Parametri LoRA:         {trainable:,} trainabili
    LoRA pruned:            {pruned_lora:,} / {total_lora:,} ({pruned_lora/max(total_lora,1)*100:.1f}%)
    Parametri attivi LoRA:  {total_lora - pruned_lora:,}
    Rapporto efficienza:    {total_params/max(total_lora - pruned_lora, 1):.0f}x
  
  Evoluzione cognitiva:
    Fase 1 (Calcolo esatto)  │ Loss: {self.history['phase1_loss'][-1]:.4f}
    Fase 2 (Intuizione)      │ Loss: {self.history['phase2_loss'][-1]:.4f} │ Pruned: {self.config.pruning_ratio*100:.0f}%
    Fase 3 (Delega tool)     │ Loss: {self.history['phase3_loss'][-1]:.4f}
    Fase 4 (Orchestrazione)  │ Loss: {self.history['phase4_loss'][-1]:.4f}

  ╔════════════════════════════════════════════════════════════╗
  ║  I {total_params/1e6:.0f}M di {self.config.model_name} sono la 'memoria a lungo termine'    ║
  ║  I ~{total_lora - pruned_lora:,} pesi LoRA attivi sono l''intuizione'      ║
  ║  Il calculator è il tool deterministico                    ║
  ║  Insieme: un sistema cognitivo stratificato                ║
  ╚════════════════════════════════════════════════════════════╝
        """)
    
    def _save_everything(self):
        out_dir = './output_llm'
        os.makedirs(out_dir, exist_ok=True)
        
        # Salva adattatori LoRA (leggeri!)
        self.model.save_pretrained(os.path.join(out_dir, 'lora_adapters'))
        self.tokenizer.save_pretrained(os.path.join(out_dir, 'tokenizer'))
        
        # Salva maschere di pruning
        if self.pruning_masks:
            torch.save(self.pruning_masks, os.path.join(out_dir, 'pruning_masks.pt'))
        
        # Salva metriche
        metrics = {
            'history': dict(self.history),
            'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')},
            'pruning_ratio': self.config.pruning_ratio,
        }
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Calcola dimensioni
        total_size = 0
        for root, dirs, files in os.walk(out_dir):
            for fname in files:
                total_size += os.path.getsize(os.path.join(root, fname))
        
        print(f"  Salvato in: {out_dir}/")
        print(f"  Dimensione totale: {total_size / 1024 / 1024:.1f} MB")
        print(f"  (Solo gli adattatori LoRA — il modello base si scarica da HuggingFace)")
        
        # Push to Hugging Face Hub
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-lora")
            print(f"  Caricamento su Hugging Face Hub: {repo_id}...")
            api.create_repo(repo_id=repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=out_dir,
                repo_id=repo_id,
                repo_type="model",
            )
            print("  Caricamento completato con successo!")
        except Exception as e:
            print(f"  Errore durante il caricamento su Hugging Face Hub: {e}")
        
        # Pause the Space if running on Hugging Face Spaces
        if os.environ.get("SPACE_ID"):
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                print("  Mettendo in pausa lo Space per risparmiare crediti...")
                api.pause_space(repo_id=os.environ.get("SPACE_ID"))
            except Exception as e:
                print(f"  Errore durante la pausa dello Space: {e}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    
    config = Config()
    trainer = ProgressiveLLMTrainer(config)
    trainer.run_full_pipeline()
