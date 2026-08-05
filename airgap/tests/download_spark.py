"""Download Spark 4.2.0 tgz with progress bar."""
import urllib.request, sys, os, tarfile

url  = "https://archive.apache.org/dist/spark/spark-4.2.0/spark-4.2.0-bin-hadoop3.tgz"
dest = sys.argv[1]  # destination path passed from bat

def hook(count, block, total):
    if total > 0:
        pct  = min(count * block * 100 // total, 100)
        done = count * block / 1024 / 1024
        tot  = total / 1024 / 1024
        bar  = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  {done:.0f}/{tot:.0f} MB", end="", flush=True)

print(f"  Downloading Spark 4.2.0...")
urllib.request.urlretrieve(url, dest, reporthook=hook)
print("\n  Verifying archive...", flush=True)
with tarfile.open(dest, "r:gz") as t:
    members = t.getmembers()
print(f"  OK — {len(members)} entries verified.")
