"""
Download and extract Apache Spark for native cluster mode.
Run this on ALL machines.
"""

import os
import sys
import urllib.request
import tarfile

SPARK_VERSION = "4.2.0"
HADOOP_VERSION = "3"
SPARK_URL = f"https://archive.apache.org/dist/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-hadoop{HADOOP_VERSION}.tgz"
INSTALL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark")
SPARK_HOME = os.path.join(INSTALL_DIR, f"spark-{SPARK_VERSION}-bin-hadoop{HADOOP_VERSION}")


def progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        mb_done = count * block_size / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)


def verify_tgz(tgz_path):
    """Quick integrity check — try to list all members."""
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            members = tar.getmembers()
        print(f"\n  Verified: {len(members)} entries in archive.")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Archive is corrupt: {e}")
        return False


def main():
    if os.path.exists(SPARK_HOME):
        print(f"Spark already installed at: {SPARK_HOME}")
        print("Delete the folder and re-run to reinstall.")
        return

    os.makedirs(INSTALL_DIR, exist_ok=True)
    tgz_path = os.path.join(INSTALL_DIR, f"spark-{SPARK_VERSION}.tgz")

    # Remove any leftover partial/corrupt download
    if os.path.exists(tgz_path):
        print(f"Removing existing (possibly corrupt) file: {tgz_path}")
        os.remove(tgz_path)

    print(f"Downloading Spark {SPARK_VERSION}...")
    print(f"  URL: {SPARK_URL}")
    urllib.request.urlretrieve(SPARK_URL, tgz_path, reporthook=progress_hook)
    print()  # newline after progress bar

    print("Verifying archive integrity...")
    if not verify_tgz(tgz_path):
        os.remove(tgz_path)
        print("Download was corrupt. Please re-run this script.")
        sys.exit(1)

    print("Extracting (this takes ~1 minute)...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(INSTALL_DIR)
    os.remove(tgz_path)

    print(f"\nSpark {SPARK_VERSION} installed at: {SPARK_HOME}")
    print(f"\nEnvironment variables (already set in run_benchmark.bat):")
    print(f"  set SPARK_HOME={SPARK_HOME}")
    print(f"  set PATH=%SPARK_HOME%\\bin;%PATH%")


if __name__ == "__main__":
    main()
