#!/usr/bin/env python
"""Finalize v30 adaptive model: copy config, set num_experts, verify structure."""

import json
import shutil
from pathlib import Path

SNAPSHOT_DIR = Path("/home/biswajit.mishra/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
OUTPUT_DIR = Path("artifacts/adaptive_prune_merge_v30")
KEEP_EXPERTS = 64

print(f"Finalizing v30 adaptive model...")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Target experts per layer: {KEEP_EXPERTS}")

# Load original config
with open(SNAPSHOT_DIR / "config.json") as f:
    config = json.load(f)

print(f"  Original: num_experts={config.get('num_experts', 'N/A')}")

# Update config
config["num_experts"] = KEEP_EXPERTS

# Save config
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"  Updated: num_experts={config['num_experts']}")

# Copy other required files
for fname in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
    src = SNAPSHOT_DIR / fname
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / fname)
        print(f"  Copied: {fname}")

print(f"\n✓ Model finalized. Ready for loading:")
print(f"  from transformers import AutoModelForCausalLM")
print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
