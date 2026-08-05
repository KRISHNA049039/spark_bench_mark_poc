"""Download Java 17 JRE portable zip with progress bar."""
import urllib.request, sys

url  = "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jre_x64_windows_hotspot_17.0.11_9.zip"
dest = sys.argv[1]

def hook(count, block, total):
    if total > 0:
        pct  = min(count * block * 100 // total, 100)
        done = count * block / 1024 / 1024
        tot  = total / 1024 / 1024
        bar  = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  {done:.0f}/{tot:.0f} MB", end="", flush=True)

print("  Downloading Java 17 JRE...")
urllib.request.urlretrieve(url, dest, reporthook=hook)
print("\n  Done.")
