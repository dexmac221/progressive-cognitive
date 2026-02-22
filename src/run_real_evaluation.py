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
    print("Loading Qwen2.5-1.5B models for comparative evaluation...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    model_id = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 1. Load Progressive Model
    base_model_prog = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Try to download from HF Hub if not local
    lora_path_prog = "./local_lora/lora_adapters"
    if not os.path.exists(lora_path_prog):
        try:
            from huggingface_hub import snapshot_download
            print("Downloading Progressive LoRA from HF Hub...")
            lora_path_prog = snapshot_download(repo_id="dexmac/progressive-cognitive-lora")
        except Exception as e:
            print(f"Could not download Progressive LoRA: {e}")
            
    if os.path.exists(lora_path_prog):
        print(f"Loading Progressive LoRA weights from {lora_path_prog}...")
        model_prog = PeftModel.from_pretrained(base_model_prog, lora_path_prog)
        model_prog = model_prog.merge_and_unload()
    else:
        print("WARNING: Progressive LoRA weights not found.")
        model_prog = base_model_prog
    model_prog.eval()
    
    # 2. Load Baseline (Flat-LoRA) Model
    base_model_flat = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Try to download from HF Hub if not local
    lora_path_flat = "./baseline_lora_adapters"
    if not os.path.exists(lora_path_flat):
        try:
            from huggingface_hub import snapshot_download
            print("Downloading Flat-LoRA from HF Hub...")
            lora_path_flat = snapshot_download(repo_id="dexmac/progressive-cognitive-baseline-lora")
        except Exception as e:
            print(f"Could not download Flat-LoRA: {e}")
            
    if os.path.exists(lora_path_flat):
        print(f"Loading Flat-LoRA weights from {lora_path_flat}...")
        model_flat = PeftModel.from_pretrained(base_model_flat, lora_path_flat)
        model_flat = model_flat.merge_and_unload()
    else:
        print("WARNING: Flat-LoRA weights not found.")
        model_flat = base_model_flat
    model_flat.eval()
    
    # 3. Load Base Model
    print("Loading Base model for comparison...")
    model_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model_base.eval()
    
    # Configure test (20 samples per category to keep it fast, ~5 mins)
    config = TestConfig(n_samples_per_test=20, device=device, max_new_tokens=100)
    
    print("Generating test suite...")
    suite = TestSuiteGenerator.generate_all(n=config.n_samples_per_test, seed=config.seed)
    
    total_tests = sum(len(s['tests']) for s in suite.values())
    print(f"Generated tests: {total_tests} in {len(suite)} categories\n")
    
    # Evaluate models
    print("\nEvaluating Progressive Model...")
    evaluator_prog = RealModelEvaluator('Qwen2.5-1.5B + Progressive LoRA', config, model_prog, tokenizer)
    
    print("\nEvaluating Flat-LoRA Model...")
    evaluator_flat = RealModelEvaluator('Qwen2.5-1.5B + Flat LoRA', config, model_flat, tokenizer)
    
    print("\nEvaluating Base Model...")
    evaluator_base = RealModelEvaluator('Qwen2.5-1.5B (Base)', config, model_base, tokenizer)
    
    evaluations = {
        'Qwen2.5-1.5B + Progressive LoRA': evaluator_prog.evaluate_suite(suite),
        'Qwen2.5-1.5B + Flat LoRA': evaluator_flat.evaluate_suite(suite),
        'Qwen2.5-1.5B (Base)': evaluator_base.evaluate_suite(suite)
    }
    
    # Generate report
    report = ComparativeReport(evaluations)
    report.generate()
    
    os.makedirs('./results', exist_ok=True)
    report.save('./results/qwen_cognitive_report.json')
    print("\nEvaluation completed! Results saved in results/qwen_cognitive_report.json")
    
    # Push results to Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "dexmac/progressive-cognitive-results")
    
    if hf_token:
        print(f"\nPushing results to Hugging Face Hub: {repo_id}")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj='./results/qwen_cognitive_report.json',
                path_in_repo="qwen_cognitive_report.json",
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token
            )
            print("Successfully pushed results to Hub!")
            
            # Pause the space to save money
            space_id = os.environ.get("SPACE_ID")
            if space_id:
                print(f"Pausing space {space_id} to save resources...")
                api.pause_space(repo_id=space_id)
        except Exception as e:
            print(f"Error pushing to hub: {e}")

if __name__ == "__main__":
    main()
