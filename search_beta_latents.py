#!/usr/bin/env python3
"""Search M4 for beta sweep artifacts (latents, models, configs).
Step 1: exhaustive listing under known BASE.
Step 2: pattern matching on names and content.
"""
import os
import json

BASE = "/Users/regis/dev/epistemologue/output"

print("=" * 60)
print("BETA SWEEP ARTIFACT SEARCH")
print("=" * 60)

# Step 1: List ALL .npy and .pt files under BASE with size and mtime
print("\n--- Step 1: All .npy and .pt under BASE ---")
artifacts = []
for root, dirs, files in os.walk(BASE):
    depth = root.replace(BASE, "").count(os.sep)
    if depth > 5:
        continue
    for f in files:
        if f.endswith((".npy", ".pt", ".pth")):
            path = os.path.join(root, f)
            sz = os.path.getsize(path)
            artifacts.append((path, sz))

print(f"  Total: {len(artifacts)} files")

# Step 2: Filter by patterns suggesting beta sweep
print("\n--- Step 2: Files matching beta/sweep/kl patterns ---")
patterns = ["beta", "b0.", "b0_", "b1", "b2", "b5", "b10", "kl", "sweep"]
matches = []
for path, sz in artifacts:
    fname = os.path.basename(path).lower()
    dname = os.path.dirname(path).lower()
    for p in patterns:
        if p in fname or p in dname:
            matches.append((path, sz, p))
            break

if matches:
    for path, sz, pat in matches[:30]:
        print(f"  [{pat}] {path} ({sz/1024:.0f} KB)")
else:
    print("  (no pattern matches in filenames/dirs)")

# Step 3: List all subdirectories under BASE (structure overview)
print("\n--- Step 3: Directory structure under BASE ---")
for root, dirs, files in os.walk(BASE):
    depth = root.replace(BASE, "").count(os.sep)
    if depth > 3:
        continue
    npy_count = sum(1 for f in files if f.endswith(".npy"))
    pt_count = sum(1 for f in files if f.endswith((".pt", ".pth")))
    if npy_count or pt_count or depth <= 2:
        indent = "  " * depth
        print(f"  {indent}{os.path.basename(root)}/ ({npy_count} npy, {pt_count} pt)")

# Step 4: Check JSON/config files for beta values
print("\n--- Step 4: Config/JSON files mentioning beta values ---")
for root, dirs, files in os.walk(BASE):
    depth = root.replace(BASE, "").count(os.sep)
    if depth > 4:
        continue
    for f in files:
        if f.endswith((".json", ".yaml", ".yml", ".cfg")):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read(5000)
                if any(x in content for x in ["beta", "kl_weight", "beta_sweep"]):
                    print(f"  {path}")
                    # Show relevant lines
                    for line in content.split("\n"):
                        if any(x in line.lower() for x in ["beta", "kl"]):
                            print(f"    {line.strip()[:100]}")
            except Exception:
                pass

# Step 5: Check models directory specifically
print("\n--- Step 5: Models directory ---")
models_dir = os.path.join(BASE, "experiment", "models")
if os.path.exists(models_dir):
    files = sorted(os.listdir(models_dir))
    print(f"  {len(files)} files in {models_dir}")
    for f in files[:20]:
        sz = os.path.getsize(os.path.join(models_dir, f))
        print(f"    {f} ({sz/1024:.0f} KB)")
    if len(files) > 20:
        print(f"    ... ({len(files) - 20} more)")
else:
    print(f"  {models_dir} does not exist")

print("\nMETRIC:search_complete=1")
