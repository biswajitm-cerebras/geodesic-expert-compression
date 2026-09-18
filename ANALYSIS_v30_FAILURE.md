# v30 Performance Failure Root Cause Analysis

## Problem
v30 evaluation showed extremely poor performance (MMLU 23.66% vs baseline 80.1%).

## Root Cause (FIXED)
**Critical bug in gate weight handling during pruning:**

1. **Original logic error**: When experts were pruned/merged, the router gate weights were not properly synchronized:
   - For **pruned layers**: Experts 0-63 were selected by saliency (e.g., original experts [50, 23, 45, 1, ...])
   - But gate weights were just sliced to [:64] (keeping original experts 0-63)
   - **Mismatch**: Router sends expert 50 → gate weight row 0 (wrong!)
   - This caused routing to completely fail

2. **Order of operations bug**: Gate weight reordering happened AFTER expert compaction for merged layers
   - For merged layers, experts at [0, 2, 4, ..., 126] hadn't been compacted yet
   - Trying to index gate[sparse_positions] when gate was already pruned → IndexError

## Solution Applied
1. **Reorder gate weights BEFORE expert compaction**
   - For pruned layers: gate columns = selected expert indices (reordered)
   - For merged layers: gate columns = sparse positions [0, 2, 4, ..., 126]

2. **Then compact merged experts** from sparse → dense positions [0, 1, 2, ..., 63]

3. **Then remove unused experts** (128 → 64 total)

### Code Changes in scripts/adaptive_prune_merge.py
- Store keep_indices for each layer during merge/prune operations
- Reorder gate weights FIRST using keep_indices
- Compact merged experts SECOND (expert weights only, gate already reordered)
- Remove unused experts THIRD
- Save checkpoint

## Verification
- Gate weight shape: [64, 2048] ✓
- Expert indices: 0-63 only ✓
- Config num_experts: 64 ✓
- Model saved successfully ✓

## Expected Performance with Fix
- Routing should work correctly now
- Gate weights properly aligned with expert indices
- Expected performance closer to paper REAP baseline (HE 91.7%, not 69.5%)
