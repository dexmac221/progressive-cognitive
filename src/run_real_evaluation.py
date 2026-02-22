import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json
import os
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

def main():
    print("Caricamento del modello Qwen2.5-1.5B e dei pesi LoRA...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapters and merge them into the base model
    lora_path = "./local_lora/lora_adapters"
    if os.path.exists(lora_path):
        print(f"Caricamento pesi LoRA da {lora_path} e fusione con il modello base...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload() # Merge weights for actual inference
    else:
        print("ATTENZIONE: Pesi LoRA non trovati, uso il modello base.")
        model = base_model
        
    model.eval()
    print("Modello con LoRA caricato con successo!\n")
    
    # Load a fresh base model for comparison
    print("Caricamento di un nuovo modello base per il confronto...")
    base_model_only = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    base_model_only.eval()
    print("Modello base caricato con successo!\n")
    
    # Configure test (20 samples per category to keep it fast, ~5 mins)
    config = TestConfig(n_samples_per_test=20, device=device, max_new_tokens=100)
    
    print("Generazione suite di test...")
    suite = TestSuiteGenerator.generate_all(n=config.n_samples_per_test, seed=config.seed)
    
    total_tests = sum(len(s['tests']) for s in suite.values())
    print(f"Test generati: {total_tests} in {len(suite)} categorie\n")
    
    # Evaluate the real model (with LoRA)
    evaluator_lora = RealModelEvaluator('Qwen2.5-1.5B + LoRA Progressivo', config, model, tokenizer)
    
    # Evaluate the base model (without LoRA)
    print("\nValutazione del modello base (senza LoRA)...")
    evaluator_base = RealModelEvaluator('Qwen2.5-1.5B (Base)', config, base_model_only, tokenizer)
    
    evaluations = {
        'Qwen2.5-1.5B + LoRA Progressivo': evaluator_lora.evaluate_suite(suite),
        'Qwen2.5-1.5B (Base)': evaluator_base.evaluate_suite(suite)
    }
    
    # Generate report
    report = ComparativeReport(evaluations)
    report.generate()
    report.save('./qwen_cognitive_report.json')
    print("\nValutazione completata! Risultati salvati in qwen_cognitive_report.json")

if __name__ == "__main__":
    main()
