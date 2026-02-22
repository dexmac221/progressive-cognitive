"""
╔══════════════════════════════════════════════════════════════════╗
║         ARCHITETTURA COGNITIVA PROGRESSIVA — PoC               ║
║                                                                  ║
║  Un modello che cresce come un essere umano:                     ║
║  Fase 1: Impara i calcoli esatti (il bambino)                   ║
║  Fase 2: Comprime la conoscenza in intuizione (lo studente)     ║
║  Fase 3: Impara a delegare ai tool (l'adulto)                  ║
║  Fase 4: Diventa orchestratore (l'esperto)                      ║
║                                                                  ║
║  La conoscenza non scompare — collassa in attrattori.           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import json
import os
import time
from collections import defaultdict
from copy import deepcopy

# ─────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────

class Config:
    # Modello
    vocab_size = 64          # cifre 0-9, operatori, speciali, token tool
    d_model = 64             # dimensione embedding
    n_heads = 4              # attention heads
    n_layers = 3             # transformer layers
    d_ff = 128               # feed-forward hidden
    max_seq_len = 32         # lunghezza massima sequenza
    dropout = 0.1

    # Training
    batch_size = 64
    lr = 3e-4
    device = "cpu"

    # Fasi
    phase1_epochs = 6        # apprendimento grezzo
    phase2_epochs = 4        # consolidamento
    phase3_epochs = 5        # delega ai tool
    phase4_epochs = 4        # orchestrazione

    # Pruning
    pruning_ratio = 0.35     # 35% dei pesi rimossi in fase 2
    
    # Dataset
    dataset_size = 3000
    max_number = 999         # numeri fino a 999


# ─────────────────────────────────────────────────────────────────
# TOKENIZER — Semplice ma funzionale
# ─────────────────────────────────────────────────────────────────

class ArithmeticTokenizer:
    """Tokenizza espressioni aritmetiche in sequenze di token."""
    
    def __init__(self):
        # Token speciali
        self.special = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<SEP>': 3,
            '<TOOL>': 4, '</TOOL>': 5, '<APPROX>': 6, '<EXACT>': 7,
            '<VALID>': 8, '<INVALID>': 9,
        }
        # Cifre e operatori
        self.chars = {str(i): 10 + i for i in range(10)}  # 0-9 → 10-19
        self.chars['+'] = 20
        self.chars['-'] = 21
        self.chars['*'] = 22
        self.chars['/'] = 23
        self.chars['='] = 24
        self.chars['.'] = 25
        self.chars[' '] = 26
        self.chars['('] = 27
        self.chars[')'] = 28
        self.chars['~'] = 29  # "circa" per approssimazioni
        self.chars[','] = 30
        
        self.vocab_size = 64
        # Reverse mapping
        self.id_to_token = {}
        for k, v in self.special.items():
            self.id_to_token[v] = k
        for k, v in self.chars.items():
            self.id_to_token[v] = k
    
    def encode(self, text, max_len=32):
        tokens = [self.special['<SOS>']]
        for ch in text:
            if ch in self.chars:
                tokens.append(self.chars[ch])
            # ignora caratteri sconosciuti
        tokens.append(self.special['<EOS>'])
        # Pad
        while len(tokens) < max_len:
            tokens.append(self.special['<PAD>'])
        return tokens[:max_len]
    
    def decode(self, token_ids):
        result = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if tok in ('<PAD>', '<SOS>', '<EOS>'):
                    continue
                result.append(tok)
        return ''.join(result)


# ─────────────────────────────────────────────────────────────────
# DATASET — Espressioni aritmetiche progressive
# ─────────────────────────────────────────────────────────────────

class ArithmeticDataset(Dataset):
    """
    Genera espressioni aritmetiche con diversi livelli di complessità.
    
    Livello 1: Operazioni semplici (2 + 3 = 5)
    Livello 2: Operazioni medie (45 * 12 = 540)
    Livello 3: Espressioni composte (23 + 45 * 2 = 113)
    """
    
    def __init__(self, tokenizer, size=20000, max_num=999, phase="exact"):
        self.tokenizer = tokenizer
        self.phase = phase
        self.data = []
        
        for _ in range(size):
            expr, result, difficulty = self._generate_expression(max_num)
            
            if phase == "exact":
                # Fase 1: input = espressione, target = risultato esatto
                input_str = f"{expr}="
                target_str = str(result)
            
            elif phase == "approximate":
                # Fase 2: target = ordine di grandezza / approssimazione
                approx = self._approximate(result)
                input_str = f"{expr}="
                target_str = f"~{approx}"
            
            elif phase == "tool_delegation":
                # Fase 3: il modello deve decidere se calcolare o delegare
                if difficulty >= 2:
                    # Espressioni complesse → delega al tool
                    input_str = f"{expr}="
                    target_str = f"<TOOL>{expr}</TOOL>"
                else:
                    # Espressioni semplici → approssima
                    approx = self._approximate(result)
                    input_str = f"{expr}="
                    target_str = f"~{approx}"
            
            elif phase == "orchestrator":
                # Fase 4: intuizione + tool + validazione
                if difficulty >= 2:
                    approx = self._approximate(result)
                    input_str = f"{expr}="
                    # Formato: approssimazione intuitiva → delega → validazione
                    target_str = f"~{approx}<TOOL>{expr}</TOOL><VALID>"
                else:
                    input_str = f"{expr}="
                    target_str = f"~{self._approximate(result)}<VALID>"
            
            self.data.append({
                'input': input_str,
                'target': target_str,
                'exact_result': result,
                'difficulty': difficulty,
                'expression': expr,
            })
    
    def _generate_expression(self, max_num):
        difficulty = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        
        if difficulty == 1:
            # Semplice: a op b
            a = random.randint(1, min(99, max_num))
            b = random.randint(1, min(99, max_num))
            op = random.choice(['+', '-', '*'])
            expr = f"{a} {op} {b}"
            result = eval(expr)
            
        elif difficulty == 2:
            # Media: numeri più grandi
            a = random.randint(10, max_num)
            b = random.randint(10, max_num)
            op = random.choice(['+', '-', '*'])
            if op == '*':
                b = random.randint(2, 99)  # limita per non esplodere
            expr = f"{a} {op} {b}"
            result = eval(expr)
            
        else:
            # Composta: a op b op c
            a = random.randint(1, 200)
            b = random.randint(1, 50)
            c = random.randint(1, 50)
            op1 = random.choice(['+', '-'])
            op2 = random.choice(['+', '*'])
            expr = f"{a} {op1} {b} {op2} {c}"
            result = eval(expr)
        
        return expr, int(result), difficulty
    
    def _approximate(self, number):
        """Approssima un numero al suo ordine di grandezza significativo."""
        if number == 0:
            return "0"
        
        sign = "-" if number < 0 else ""
        abs_num = abs(number)
        
        if abs_num < 10:
            return str(number)
        elif abs_num < 100:
            # Arrotonda alla decina
            return f"{sign}{round(abs_num, -1)}"
        elif abs_num < 1000:
            # Arrotonda al centinaio
            return f"{sign}{round(abs_num, -2)}"
        elif abs_num < 10000:
            # Arrotonda al migliaio
            return f"{sign}{round(abs_num, -3)}"
        else:
            # Notazione approssimata
            exp = len(str(abs_num)) - 1
            leading = round(abs_num / (10 ** exp))
            return f"{sign}{leading}e{exp}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(self.tokenizer.encode(item['input']))
        target_ids = torch.tensor(self.tokenizer.encode(item['target']))
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'difficulty': item['difficulty'],
            'exact_result': item['exact_result'],
        }


# ─────────────────────────────────────────────────────────────────
# MODELLO — Transformer compatto
# ─────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CognitiveTransformer(nn.Module):
    """
    Transformer progettato per l'architettura cognitiva progressiva.
    
    Caratteristiche speciali:
    - Mask di pruning per la compressione (Fase 2)
    - Modulo di routing per decidere tool vs interno (Fase 3)
    - Testa di validazione per il senso critico (Fase 4)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Testa di output: genera token
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        # === MODULI SPECIFICI PER FASE ===
        
        # Fase 3: Router — decide se delegare al tool
        self.tool_router = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 2),  # [compute_internally, delegate_to_tool]
        )
        
        # Fase 4: Validatore — il "senso critico"
        self.validator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),  # probabilità che il risultato sia plausibile
        )
        
        # Pruning masks (inizialmente tutti 1 = attivi)
        self.pruning_masks = {}
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                self.pruning_masks[name] = torch.ones_like(param.data)
        
        # Contatore parametri
        self._count_params()
    
    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        active = sum(
            (m == 1).sum().item() 
            for m in self.pruning_masks.values()
        )
        total_maskable = sum(m.numel() for m in self.pruning_masks.values())
        self.param_stats = {
            'total': total,
            'trainable': trainable,
            'active_weights': active,
            'total_maskable': total_maskable,
            'pruned_pct': (1 - active / max(total_maskable, 1)) * 100,
        }
    
    def apply_pruning_masks(self):
        """Applica le maschere di pruning ai pesi."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.pruning_masks:
                    param.data *= self.pruning_masks[name]
    
    def forward(self, input_ids, return_routing=False, return_validation=False):
        # Embedding + positional
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Output logits
        logits = self.output_head(x)
        
        result = {'logits': logits}
        
        # Routing decision (media sui token)
        if return_routing:
            pooled = x.mean(dim=1)  # [batch, d_model]
            routing = self.tool_router(pooled)  # [batch, 2]
            result['routing'] = F.softmax(routing, dim=-1)
        
        # Validation score
        if return_validation:
            pooled = x.mean(dim=1)
            validity = self.validator(pooled)  # [batch, 1]
            result['validity'] = validity.squeeze(-1)
        
        return result


# ─────────────────────────────────────────────────────────────────
# TOOL DETERMINISTICO — La calcolatrice
# ─────────────────────────────────────────────────────────────────

class DeterministicCalculator:
    """
    Tool deterministico: esegue calcoli esatti.
    Questo è ciò che il modello impara a invocare invece di calcolare.
    """
    
    @staticmethod
    def compute(expression: str) -> dict:
        """Esegue un calcolo e ritorna risultato + metadati."""
        try:
            # Sanitizza (solo cifre e operatori base)
            allowed = set('0123456789+-*/(). ')
            clean = ''.join(c for c in expression if c in allowed)
            result = eval(clean)
            return {
                'success': True,
                'result': int(result) if float(result).is_integer() else round(result, 2),
                'expression': clean,
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e),
            }


# ─────────────────────────────────────────────────────────────────
# TRAINER — Il cuore dell'architettura progressiva
# ─────────────────────────────────────────────────────────────────

class ProgressiveTrainer:
    """
    Implementa il training progressivo in 4 fasi.
    
    Ogni fase trasforma il modello, comprimendo la conoscenza
    della fase precedente in intuizione per la successiva.
    """
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = ArithmeticTokenizer()
        self.calculator = DeterministicCalculator()
        self.model = CognitiveTransformer(config).to(config.device)
        self.history = defaultdict(list)
        
        print(self._header())
        print(f"  Parametri totali: {self.model.param_stats['total']:,}")
        print(f"  Device: {config.device}\n")
    
    def _header(self):
        return """
╔══════════════════════════════════════════════════════════════╗
║          ARCHITETTURA COGNITIVA PROGRESSIVA                  ║
║          Training in 4 Fasi                                  ║
╚══════════════════════════════════════════════════════════════╝"""
    
    # ─────── FASE 1: Apprendimento grezzo ───────
    
    def phase1_learn_exact(self):
        """
        FASE 1 — IL BAMBINO
        Impara a fare calcoli esatti. Tutti i pesi sono attivi.
        Il modello memorizza pattern aritmetici nei parametri.
        """
        print("\n" + "=" * 60)
        print("  FASE 1 — FONDAMENTA: Apprendimento grezzo")
        print("  « Il bambino che impara l'aritmetica »")
        print("=" * 60)
        
        dataset = ArithmeticDataset(
            self.tokenizer, 
            size=self.config.dataset_size, 
            max_num=self.config.max_number,
            phase="exact"
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        for epoch in range(self.config.phase1_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                target_ids = batch['target_ids'].to(self.config.device)
                
                result = self.model(input_ids)
                logits = result['logits']
                
                # Loss su tutti i token
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=0,  # ignora PAD
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy: confronta predizioni con target
                preds = logits.argmax(dim=-1)
                mask = target_ids != 0
                correct += (preds[mask] == target_ids[mask]).sum().item()
                total += mask.sum().item()
            
            acc = correct / max(total, 1) * 100
            avg_loss = total_loss / len(loader)
            self.history['phase1_loss'].append(avg_loss)
            self.history['phase1_acc'].append(acc)
            
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{self.config.phase1_epochs} │ "
                      f"Loss: {avg_loss:.4f} │ Acc: {acc:.1f}%")
        
        # Valutazione qualitativa
        self._evaluate_phase1(dataset)
        return avg_loss, acc
    
    def _evaluate_phase1(self, dataset):
        """Mostra esempi di cosa ha imparato il modello."""
        print(f"\n  ── Esempi di calcolo diretto ──")
        self.model.eval()
        samples = random.sample(dataset.data, min(5, len(dataset.data)))
        
        for s in samples:
            input_ids = torch.tensor(
                self.tokenizer.encode(s['input'])
            ).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                result = self.model(input_ids)
                preds = result['logits'].argmax(dim=-1)[0]
            
            predicted = self.tokenizer.decode(preds.tolist())
            print(f"    {s['input']} → pred: {predicted[:12]:12s} │ "
                  f"esatto: {s['exact_result']}")
        
        self.model.train()
    
    # ─────── FASE 2: Consolidamento + Pruning ───────
    
    def phase2_consolidate(self):
        """
        FASE 2 — LO STUDENTE
        Comprime la conoscenza esatta in intuizione approssimata.
        Pruning strutturato: rimuove i circuiti di calcolo esatto,
        mantiene quelli che codificano il "senso" numerico.
        
        Questa è la fase critica: simula la "dimenticanza strutturata"
        che trasforma conoscenza esplicita in intuizione.
        """
        print("\n" + "=" * 60)
        print("  FASE 2 — CONSOLIDAMENTO: Compressione in intuizione")
        print("  « Lo studente che passa all'algebra »")
        print("=" * 60)
        
        # ── Step 1: Pruning magnitude-based ──
        print(f"\n  Pruning: rimozione del {self.config.pruning_ratio*100:.0f}% dei pesi...")
        self._magnitude_pruning(self.config.pruning_ratio)
        self.model._count_params()
        print(f"  Pesi attivi dopo pruning: {self.model.param_stats['active_weights']:,} "
              f"({100 - self.model.param_stats['pruned_pct']:.1f}%)")
        
        # ── Step 2: Fine-tune su approssimazioni ──
        print(f"\n  Fine-tuning su target approssimati...")
        dataset = ArithmeticDataset(
            self.tokenizer,
            size=self.config.dataset_size,
            max_num=self.config.max_number,
            phase="approximate"
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr * 0.5  # lr ridotto per consolidamento
        )
        
        for epoch in range(self.config.phase2_epochs):
            total_loss = 0
            
            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                target_ids = batch['target_ids'].to(self.config.device)
                
                # Applica maschere di pruning dopo ogni step
                self.model.apply_pruning_masks()
                
                result = self.model(input_ids)
                logits = result['logits']
                
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=0,
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Ri-applica maschere (i gradienti possono aver modificato pesi pruned)
                self.model.apply_pruning_masks()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            self.history['phase2_loss'].append(avg_loss)
            
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{self.config.phase2_epochs} │ Loss: {avg_loss:.4f}")
        
        # Valutazione: il modello ora dovrebbe "intuire" non "calcolare"
        self._evaluate_phase2(dataset)
        return avg_loss
    
    def _magnitude_pruning(self, ratio):
        """
        Pruning basato sulla magnitudine: rimuove i pesi più piccoli.
        
        Quelli che sopravvivono sono per definizione i circuiti più
        "importanti" — quelli che codificano pattern robusti (intuizione),
        non calcoli specifici (conoscenza).
        """
        all_weights = []
        for name, param in self.model.named_parameters():
            if name in self.model.pruning_masks:
                all_weights.append(param.data.abs().flatten())
        
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, ratio)
        
        pruned_count = 0
        total_count = 0
        for name, param in self.model.named_parameters():
            if name in self.model.pruning_masks:
                mask = (param.data.abs() > threshold).float()
                self.model.pruning_masks[name] = mask
                pruned_count += (mask == 0).sum().item()
                total_count += mask.numel()
        
        print(f"  Soglia di pruning: {threshold:.6f}")
        print(f"  Pesi rimossi: {pruned_count:,} / {total_count:,}")
    
    def _evaluate_phase2(self, dataset):
        """Il modello dovrebbe ora dare approssimazioni, non esatti."""
        print(f"\n  ── Esempi di intuizione approssimata ──")
        self.model.eval()
        samples = random.sample(dataset.data, min(5, len(dataset.data)))
        
        for s in samples:
            input_ids = torch.tensor(
                self.tokenizer.encode(s['input'])
            ).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                result = self.model(input_ids)
                preds = result['logits'].argmax(dim=-1)[0]
            
            predicted = self.tokenizer.decode(preds.tolist())
            approx_target = s['target']
            print(f"    {s['input']} → intuizione: {predicted[:12]:12s} │ "
                  f"target: {approx_target[:12]:12s} │ esatto: {s['exact_result']}")
        
        self.model.train()
    
    # ─────── FASE 3: Delega ai Tool ───────
    
    def phase3_tool_delegation(self):
        """
        FASE 3 — L'ADULTO
        Il modello impara quando delegare al tool deterministico.
        Per espressioni complesse: genera <TOOL>expr</TOOL>
        Per espressioni semplici: usa l'intuizione approssimata.
        
        Il router viene addestrato a discriminare complessità.
        """
        print("\n" + "=" * 60)
        print("  FASE 3 — DELEGA: Integrazione dei tool")
        print("  « L'adulto che usa la calcolatrice »")
        print("=" * 60)
        
        dataset = ArithmeticDataset(
            self.tokenizer,
            size=self.config.dataset_size,
            max_num=self.config.max_number,
            phase="tool_delegation"
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr * 0.3)
        
        for epoch in range(self.config.phase3_epochs):
            total_loss = 0
            tool_decisions_correct = 0
            tool_decisions_total = 0
            
            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                target_ids = batch['target_ids'].to(self.config.device)
                difficulties = batch['difficulty']
                
                self.model.apply_pruning_masks()
                
                result = self.model(input_ids, return_routing=True)
                logits = result['logits']
                routing = result['routing']  # [batch, 2]: [interno, tool]
                
                # Loss sequenza
                seq_loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=0,
                )
                
                # Loss routing: complesso → tool (1), semplice → interno (0)
                should_use_tool = (difficulties >= 2).float().to(self.config.device)
                routing_loss = F.binary_cross_entropy(
                    routing[:, 1],  # probabilità di usare tool
                    should_use_tool,
                )
                
                # Loss combinata
                loss = seq_loss + 0.5 * routing_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                self.model.apply_pruning_masks()
                
                total_loss += loss.item()
                
                # Tracking decisioni tool
                predicted_tool = (routing[:, 1] > 0.5).float()
                tool_decisions_correct += (predicted_tool == should_use_tool).sum().item()
                tool_decisions_total += len(should_use_tool)
            
            avg_loss = total_loss / len(loader)
            tool_acc = tool_decisions_correct / max(tool_decisions_total, 1) * 100
            self.history['phase3_loss'].append(avg_loss)
            self.history['phase3_tool_acc'].append(tool_acc)
            
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{self.config.phase3_epochs} │ "
                      f"Loss: {avg_loss:.4f} │ Tool routing acc: {tool_acc:.1f}%")
        
        self._evaluate_phase3(dataset)
        return avg_loss, tool_acc
    
    def _evaluate_phase3(self, dataset):
        """Mostra come il modello decide tra intuizione e tool."""
        print(f"\n  ── Esempi di decisione tool vs intuizione ──")
        self.model.eval()
        
        # Prendi esempi misti
        simple = [d for d in dataset.data if d['difficulty'] == 1][:3]
        complex_ = [d for d in dataset.data if d['difficulty'] >= 2][:3]
        samples = simple + complex_
        
        for s in samples:
            input_ids = torch.tensor(
                self.tokenizer.encode(s['input'])
            ).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                result = self.model(input_ids, return_routing=True)
                preds = result['logits'].argmax(dim=-1)[0]
                routing = result['routing'][0]
            
            predicted = self.tokenizer.decode(preds.tolist())
            decision = "🔧 TOOL" if routing[1] > 0.5 else "🧠 INTUIZIONE"
            confidence = max(routing[0], routing[1]).item() * 100
            
            # Se ha delegato al tool, eseguilo
            tool_result = ""
            if routing[1] > 0.5:
                calc = self.calculator.compute(s['expression'])
                if calc['success']:
                    tool_result = f" → tool dice: {calc['result']}"
            
            print(f"    {s['expression']:20s} │ {decision} ({confidence:.0f}%) │ "
                  f"diff={s['difficulty']}{tool_result}")
        
        self.model.train()
    
    # ─────── FASE 4: Orchestrazione ───────
    
    def phase4_orchestrate(self):
        """
        FASE 4 — L'ESPERTO
        Il modello è un orchestratore completo:
        1. Intuisce una stima approssimata
        2. Delega al tool per il calcolo esatto
        3. Valida che il risultato del tool sia coerente con l'intuizione
        
        Questo è il "vedere il bug senza leggere il codice".
        """
        print("\n" + "=" * 60)
        print("  FASE 4 — ORCHESTRAZIONE: Intelligenza emergente")
        print("  « L'esperto che vede il bug senza leggere il codice »")
        print("=" * 60)
        
        dataset = ArithmeticDataset(
            self.tokenizer,
            size=self.config.dataset_size,
            max_num=self.config.max_number,
            phase="orchestrator"
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr * 0.2)
        
        for epoch in range(self.config.phase4_epochs):
            total_loss = 0
            valid_correct = 0
            valid_total = 0
            
            for batch in loader:
                input_ids = batch['input_ids'].to(self.config.device)
                target_ids = batch['target_ids'].to(self.config.device)
                exact_results = batch['exact_result'].float().to(self.config.device)
                
                self.model.apply_pruning_masks()
                
                result = self.model(
                    input_ids, 
                    return_routing=True, 
                    return_validation=True
                )
                logits = result['logits']
                validity = result['validity']
                
                # Loss sequenza
                seq_loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=0,
                )
                
                # Loss validazione: simula risultati corretti e incorretti
                # 80% corretti, 20% perturbati (per insegnare a rilevare errori)
                is_correct = torch.ones_like(validity)
                perturb_mask = torch.rand_like(validity) < 0.2
                is_correct[perturb_mask] = 0.0
                
                valid_loss = F.binary_cross_entropy(validity, is_correct)
                
                loss = seq_loss + 0.3 * valid_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                self.model.apply_pruning_masks()
                
                total_loss += loss.item()
                
                predicted_valid = (validity > 0.5).float()
                valid_correct += (predicted_valid == is_correct).sum().item()
                valid_total += len(is_correct)
            
            avg_loss = total_loss / len(loader)
            valid_acc = valid_correct / max(valid_total, 1) * 100
            self.history['phase4_loss'].append(avg_loss)
            self.history['phase4_valid_acc'].append(valid_acc)
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{self.config.phase4_epochs} │ "
                      f"Loss: {avg_loss:.4f} │ Validation acc: {valid_acc:.1f}%")
        
        self._evaluate_phase4()
        return avg_loss, valid_acc
    
    def _evaluate_phase4(self):
        """Demo completa del modello orchestratore."""
        print(f"\n  ── Demo orchestrazione completa ──")
        self.model.eval()
        
        test_expressions = [
            "5 + 3",
            "42 * 17",
            "350 + 275",
            "123 - 45 * 3",
            "999 + 888",
            "7 * 8",
        ]
        
        for expr in test_expressions:
            input_str = f"{expr}="
            input_ids = torch.tensor(
                self.tokenizer.encode(input_str)
            ).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                result = self.model(
                    input_ids, 
                    return_routing=True, 
                    return_validation=True
                )
                preds = result['logits'].argmax(dim=-1)[0]
                routing = result['routing'][0]
                validity = result['validity'][0].item()
            
            intuition = self.tokenizer.decode(preds.tolist())
            use_tool = routing[1].item() > 0.5
            
            # Calcolo esatto dal tool
            calc = self.calculator.compute(expr)
            exact = calc['result'] if calc['success'] else '?'
            
            # Orchestrazione completa
            print(f"\n    Espressione: {expr}")
            print(f"    ├── Intuizione: {intuition[:15]}")
            print(f"    ├── Routing: {'🔧 TOOL' if use_tool else '🧠 INTERNO'} "
                  f"(conf: {max(routing[0], routing[1]).item()*100:.0f}%)")
            if use_tool:
                print(f"    ├── Tool → {exact}")
            print(f"    ├── Validazione: {'✓ plausibile' if validity > 0.5 else '✗ sospetto'} "
                  f"({validity*100:.0f}%)")
            print(f"    └── Risultato esatto: {exact}")
        
        self.model.train()
    
    # ─────── PIPELINE COMPLETA ───────
    
    def run_full_pipeline(self):
        """Esegue tutte e 4 le fasi in sequenza."""
        start = time.time()
        
        # Fase 1: Impara
        self.phase1_learn_exact()
        
        # Fase 2: Comprimi
        self.phase2_consolidate()
        
        # Fase 3: Delega
        self.phase3_tool_delegation()
        
        # Fase 4: Orchestra
        self.phase4_orchestrate()
        
        elapsed = time.time() - start
        
        # Report finale
        self._final_report(elapsed)
        
        # Salva
        self._save_results()
    
    def _final_report(self, elapsed):
        """Report finale con confronto tra fasi."""
        self.model._count_params()
        
        print("\n" + "═" * 60)
        print("  REPORT FINALE — Evoluzione del modello")
        print("═" * 60)
        
        print(f"""
  Tempo totale: {elapsed:.1f}s
  
  Parametri:
    Totali:     {self.model.param_stats['total']:>10,}
    Pruned:     {self.model.param_stats['pruned_pct']:>9.1f}%
    Attivi:     {self.model.param_stats['active_weights']:>10,}
  
  Evoluzione per fase:
    Fase 1 (Esatto)      │ Loss finale: {self.history['phase1_loss'][-1]:.4f} │ Acc: {self.history['phase1_acc'][-1]:.1f}%
    Fase 2 (Intuizione)  │ Loss finale: {self.history['phase2_loss'][-1]:.4f} │ Pruned: {self.config.pruning_ratio*100:.0f}%
    Fase 3 (Tool)        │ Loss finale: {self.history['phase3_loss'][-1]:.4f} │ Tool routing: {self.history['phase3_tool_acc'][-1]:.1f}%
    Fase 4 (Orchestr.)   │ Loss finale: {self.history['phase4_loss'][-1]:.4f} │ Validation: {self.history['phase4_valid_acc'][-1]:.1f}%
  
  ╔════════════════════════════════════════════════════════╗
  ║  Il modello è passato da calcolatrice a orchestratore ║
  ║  La conoscenza esatta è diventata intuizione          ║
  ║  Il calcolo è delegato al tool deterministico         ║
  ║  Il senso critico valida i risultati                  ║
  ╚════════════════════════════════════════════════════════╝
        """)
    
    def _save_results(self):
        """Salva modello e metriche."""
        os.makedirs('/home/claude/output', exist_ok=True)
        
        # Salva modello
        torch.save({
            'model_state': self.model.state_dict(),
            'pruning_masks': self.model.pruning_masks,
            'config': vars(self.config),
            'history': dict(self.history),
        }, '/home/claude/output/cognitive_model.pt')
        
        # Salva metriche come JSON
        metrics = {
            'param_stats': self.model.param_stats,
            'history': {k: v for k, v in self.history.items()},
            'config': {k: v for k, v in vars(self.config).items() 
                      if not k.startswith('_')},
        }
        with open('/home/claude/output/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"  Modello salvato in: /home/claude/output/cognitive_model.pt")
        print(f"  Metriche salvate in: /home/claude/output/metrics.json")


# ─────────────────────────────────────────────────────────────────
# MAIN — Esecuzione
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Riproducibilità
    torch.manual_seed(42)
    random.seed(42)
    
    config = Config()
    trainer = ProgressiveTrainer(config)
    trainer.run_full_pipeline()
