"""
Script to train the Baseline (Control) model.
This script trains Qwen2.5-1.5B with LoRA on the same data as the progressive model,
but WITHOUT the 4-phase curriculum and WITHOUT pruning.
All data is mixed and passed to the model in a single training phase.
This serves to demonstrate that the improvements come from the cognitive architecture
and not simply from fine-tuning with LoRA.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import random
import json
import time

# ─────────────────────────────────────────────────────────────────
# BASELINE CONFIGURATION
# ─────────────────────────────────────────────────────────────────

class BaselineConfig:
    model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Same LoRA configuration as the progressive model
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    
    batch_size = 4
    max_seq_len = 128
    
    # Total samples = 2000 + 1500 + 1500 + 1000 = 6000
    total_samples = 6000
    epochs = 3
    lr = 1e-4
    
    max_number = 9999

# ─────────────────────────────────────────────────────────────────
# MIXED DATA GENERATION (All phases together)
# ─────────────────────────────────────────────────────────────────

class MixedMathDataset(Dataset):
    def __init__(self, tokenizer, num_samples, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        print(f"Generating {num_samples} mixed samples for the Baseline...")
        for _ in range(num_samples):
            # Randomly choose the task type (Phase 1, 2, 3 or 4)
            task_type = random.choice(['exact', 'estimate', 'route', 'orchestrate'])
            
            # Generate expression
            difficulty = random.choices([1, 2, 3], weights=[0.35, 0.40, 0.25])[0]
            if difficulty == 1:
                a, b = random.randint(1, 99), random.randint(1, 99)
                op = random.choice(['+', '-', '*'])
            elif difficulty == 2:
                a, b = random.randint(100, 9999), random.randint(10, 999)
                op = random.choice(['+', '-', '*'])
                if op == '*': a, b = random.randint(10, 999), random.randint(2, 99)
            else:
                a, b, c = random.randint(10, 999), random.randint(10, 99), random.randint(2, 9)
                op1, op2 = random.choice(['+', '-']), '*'
                expr = f"{a} {op1} {b} {op2} {c}"
                exact_val = eval(expr)
            
            if difficulty < 3:
                expr = f"{a} {op} {b}"
                exact_val = eval(expr)
            
            # Format based on task
            if task_type == 'exact':
                text = f"Calculate: {expr} = {exact_val}"
            elif task_type == 'estimate':
                order = len(str(abs(exact_val))) - 1
                if order <= 1: est = "tens"
                elif order == 2: est = "hundreds"
                elif order == 3: est = "thousands"
                else: est = f"10^{order}"
                text = f"Estimate: {expr} = order of {est}"
            elif task_type == 'route':
                if difficulty == 1: decision = "INTERNAL CALCULATION"
                else: decision = "DELEGATE TO TOOL"
                text = f"Analyze: {expr}\nComplexity: {difficulty}\nDecision: {decision}"
            else: # orchestrate
                if difficulty == 1:
                    text = f"Solve: {expr}\nStep 1 - Intuition: exact\nStep 2 - Routing: INTERNAL\nResult: {exact_val}"
                else:
                    order = len(str(abs(exact_val))) - 1
                    text = f"Solve: {expr}\nStep 1 - Intuition: order 10^{order}\nStep 2 - Routing: DELEGATE\nStep 3 - Tool: {exact_val}\nStep 4 - Validation: consistent -> VALID\nResult: {exact_val}"
            
            self.samples.append(text)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train_baseline():
    print("Initializing Baseline Training (Control)...")
    
    tokenizer = AutoTokenizer.from_pretrained(BaselineConfig.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BaselineConfig.model_name,
        device_map=BaselineConfig.device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        r=BaselineConfig.lora_r,
        lora_alpha=BaselineConfig.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=BaselineConfig.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    dataset = MixedMathDataset(tokenizer, BaselineConfig.total_samples, BaselineConfig.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=BaselineConfig.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=BaselineConfig.lr)
    
    print(f"\nStarting Baseline training: {BaselineConfig.epochs} epochs on {BaselineConfig.total_samples} mixed samples.")
    
    model.train()
    for epoch in range(BaselineConfig.epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(BaselineConfig.device)
            attention_mask = batch["attention_mask"].to(BaselineConfig.device)
            labels = batch["labels"].to(BaselineConfig.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{BaselineConfig.epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- End Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} ---")
        
    # Save
    output_dir = "./baseline_lora_adapters"
    print(f"\nSaving Baseline model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Baseline Training completed!")

if __name__ == "__main__":
    train_baseline()
