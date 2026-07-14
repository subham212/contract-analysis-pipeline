"""
run_all.py — Cross-platform batch runner (works in PowerShell AND Git Bash).

Usage:
    python run_all.py                     # run all 50 contracts
    python run_all.py 36 5 14             # 36 contracts, batch of 5, start at offset 14
    python run_all.py 10 5 0              # first 10 contracts
"""
import subprocess
import sys

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 50
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 5
START = int(sys.argv[3]) if len(sys.argv) > 3 else 0

PY = sys.executable  # use whatever python is running this script

print("=" * 52)
print(f"  Contract Pipeline — {TOTAL} contracts, batches of {BATCH}, start offset {START}")
print("=" * 52)
print()

batch_num = START // BATCH + 1
offset = START
while offset < START + TOTAL:
    count = min(BATCH, START + TOTAL - offset)
    tag = f"batch_{batch_num:02d}"
    print(f">>> [{tag}] contracts {offset + 1}-{offset + count} of {START + TOTAL}")
    result = subprocess.run(
        [PY, "main.py", "run", "--max", str(count),
         "--offset", str(offset), "--batch", tag, "--skip-download"],
        capture_output=False,
    )
    print()
    offset += BATCH
    batch_num += 1

print("=" * 52)
print("  Merging all batches into final output...")
print("=" * 52)
subprocess.run([PY, "main.py", "merge"])
print()
print("DONE! Final files:")
print("  output/final_contracts.csv")
print("  output/final_contracts.json")
