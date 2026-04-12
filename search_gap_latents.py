#!/usr/bin/env python3
"""Search M4 for Gap Detection V1 latent files (.npy).
Quick scan — not a benchmark, just a file search."""

import subprocess
import os

print("=" * 60)
print("SEARCHING M4 FOR GAP DETECTION V1 LATENT FILES")
print("=" * 60)

# Known patterns from V1 config
patterns = [
    "full_seed42_latents.npy",
    "minus_7_seed42_latents.npy",
    "ghost_aspiration.json",
]

# Search common locations
search_dirs = [
    os.path.expanduser("~"),
    "/tmp",
    "/data" if os.path.exists("/data") else None,
]
search_dirs = [d for d in search_dirs if d]

print(f"\nSearching in: {search_dirs}")
print()

# Broad search for .npy files related to gap detection
for d in search_dirs:
    print(f"--- Searching {d} ---")
    try:
        result = subprocess.run(
            ["find", d, "-maxdepth", "6", "-name", "*latents.npy", "-type", "f"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                print(f"  FOUND: {line}")
        else:
            print(f"  (no *latents.npy found)")
    except Exception as e:
        print(f"  ERROR: {e}")

print()

# Also search for gap_detection or ghost directories
for d in search_dirs:
    print(f"--- Searching {d} for gap/ghost dirs ---")
    try:
        result = subprocess.run(
            ["find", d, "-maxdepth", "5", "-iname", "*gap*detect*", "-type", "d"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                print(f"  DIR: {line}")
                # List contents
                r2 = subprocess.run(["ls", "-la", line], capture_output=True, text=True, timeout=5)
                for l2 in r2.stdout.strip().split("\n")[:10]:
                    print(f"    {l2}")
        else:
            print(f"  (no gap_detect dirs found)")
    except Exception as e:
        print(f"  ERROR: {e}")

print()

# Search for output/ directories with latents/ subdirs
for d in search_dirs:
    print(f"--- Searching {d} for output/latents/ ---")
    try:
        result = subprocess.run(
            ["find", d, "-maxdepth", "6", "-path", "*/output/latents", "-type", "d"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                print(f"  DIR: {line}")
                r2 = subprocess.run(["ls", line], capture_output=True, text=True, timeout=5)
                count = len(r2.stdout.strip().split("\n")) if r2.stdout.strip() else 0
                print(f"    ({count} files)")
                # Show first 5
                for l2 in r2.stdout.strip().split("\n")[:5]:
                    print(f"    {l2}")
                if count > 5:
                    print(f"    ... ({count - 5} more)")
        else:
            print(f"  (no output/latents/ found)")
    except Exception as e:
        print(f"  ERROR: {e}")

print()
print("METRIC:search_complete=1")
