"""
Quick status check for multi-seed evaluation.
Run this periodically to see if results have been uploaded.
"""
from huggingface_hub import HfApi, list_repo_files
import sys
import os

TOKEN = os.environ.get('HF_TOKEN')
assert TOKEN, 'Set HF_TOKEN environment variable'
SPACE = 'dexmac/progressive-cognitive-eval-en-multiseed'
DATASET = 'dexmac/progressive-cognitive-results'

api = HfApi(token=TOKEN)

# 1. Space status
info = api.space_info(repo_id=SPACE)
print(f"🔄 Space status: {info.runtime.stage}")
print(f"   Hardware: {info.runtime.hardware}")

# 2. Check for new results in dataset
try:
    files = list_repo_files(DATASET, repo_type='dataset')
    seed_files = [f for f in files if 'seed' in f]
    print(f"\n📊 Seed result files in dataset repo:")
    if seed_files:
        for f in sorted(seed_files):
            print(f"   ✅ {f}")
    else:
        print("   (none yet — evaluation still running)")
    
    other_files = [f for f in files if 'seed' not in f and not f.startswith('.')]
    if other_files:
        print(f"\n   Other files: {', '.join(other_files)}")
except Exception as e:
    print(f"   Error checking dataset: {e}")

# 3. Expected files
expected = [
    'english_eval_seed42_report.json',
    'english_eval_seed42_summary.json',
    'english_eval_seed43_report.json',
    'english_eval_seed43_summary.json',
    'english_eval_seed44_report.json',
    'english_eval_seed44_summary.json',
]
existing = set(seed_files) if seed_files else set()
missing = [f for f in expected if f not in existing]
if missing:
    print(f"\n⏳ Still waiting for: {', '.join(missing)}")
else:
    print(f"\n🎉 ALL seed results uploaded! Ready for aggregation.")

if info.runtime.stage == 'PAUSED':
    print("\n💤 Space is PAUSED — evaluation likely completed!")
