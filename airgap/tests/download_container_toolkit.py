"""
Download NVIDIA Container Toolkit RPM packages for RHEL offline install.

NVIDIA's repo structure changed — individual RPM URLs are no longer stable.
The correct approach is to download the .repo file and use dnf/yum download
on a connected RHEL machine to pull exact packages with all dependencies.

This script downloads:
  1. The .repo config file (for reference)
  2. The 2 libnvidia-container RPMs (these still have direct URLs)
  3. Writes a helper script for downloading toolkit RPMs on a connected RHEL VM

Usage: python download_container_toolkit.py <output_dir>
"""
import urllib.request, sys, os

dest_dir = sys.argv[1]
os.makedirs(dest_dir, exist_ok=True)

REPO_URL = "https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo"

# libnvidia-container packages still have direct RPM paths
DIRECT_RPMS = [
    {
        "name": "libnvidia-container1",
        "url": "https://nvidia.github.io/libnvidia-container/stable/rpm/x86_64/libnvidia-container1-1.17.6-1.x86_64.rpm"
    },
    {
        "name": "libnvidia-container-tools",
        "url": "https://nvidia.github.io/libnvidia-container/stable/rpm/x86_64/libnvidia-container-tools-1.17.6-1.x86_64.rpm"
    },
]

def download_file(url, dest_path):
    print(f"  Downloading {os.path.basename(dest_path)} ...", end="", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    size = os.path.getsize(dest_path)
    print(f" {size/1024:.0f} KB  OK")

print("NVIDIA Container Toolkit — RHEL offline packages")
print(f"Output: {dest_dir}")
print()

# 1. Download repo file
repo_dest = os.path.join(dest_dir, "nvidia-container-toolkit.repo")
if os.path.exists(repo_dest):
    print(f"  [SKIP] nvidia-container-toolkit.repo already exists.")
else:
    download_file(REPO_URL, repo_dest)

# 2. Download libnvidia-container RPMs (direct URL still works)
for pkg in DIRECT_RPMS:
    fname = os.path.basename(pkg["url"])
    dest  = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        print(f"  [SKIP] {fname} already exists.")
    else:
        download_file(pkg["url"], dest)

# 3. Write a helper script to run on a connected RHEL machine
#    to download the toolkit RPMs with all dependencies
helper_script = os.path.join(dest_dir, "fetch_toolkit_rpms_on_rhel.sh")
with open(helper_script, "w", newline="\n") as f:
    f.write("""#!/bin/bash
# Run this on a RHEL 8/9 machine WITH internet access.
# It downloads nvidia-container-toolkit and all its RPM dependencies
# into ./toolkit_rpms/ ready for offline install on the airgapped machine.
#
# Usage:
#   chmod +x fetch_toolkit_rpms_on_rhel.sh
#   sudo bash fetch_toolkit_rpms_on_rhel.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/toolkit_rpms"
mkdir -p "$OUT"

echo "Adding NVIDIA Container Toolkit repo..."
cp "$SCRIPT_DIR/nvidia-container-toolkit.repo" /etc/yum.repos.d/
dnf makecache

echo "Downloading toolkit RPMs with all dependencies..."
dnf download --resolve --destdir="$OUT" nvidia-container-toolkit

echo ""
echo "Downloaded to: $OUT"
ls -lh "$OUT"
echo ""
echo "Copy $OUT/*.rpm to the airgapped machine, then run:"
echo "  sudo rpm -ivh --nodeps *.rpm"
echo "  sudo nvidia-ctk runtime configure --runtime=docker"
echo "  sudo systemctl restart docker"
""")
print(f"  fetch_toolkit_rpms_on_rhel.sh written.")

# 4. Write the offline install script
install_script = os.path.join(dest_dir, "install_offline.sh")
with open(install_script, "w", newline="\n") as f:
    f.write("""#!/bin/bash
# Install NVIDIA Container Toolkit offline on RHEL 8/9 airgapped machine.
# Run as root: sudo bash install_offline.sh
#
# Place all *.rpm files in the same folder as this script.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing RPMs from: $SCRIPT_DIR"
RPM_COUNT=$(ls "$SCRIPT_DIR"/*.rpm 2>/dev/null | wc -l)
if [ "$RPM_COUNT" -eq 0 ]; then
    echo "ERROR: No RPM files found in $SCRIPT_DIR"
    echo "Run fetch_toolkit_rpms_on_rhel.sh on a connected RHEL machine first"
    echo "and copy the resulting toolkit_rpms/*.rpm files here."
    exit 1
fi

echo "Found $RPM_COUNT RPM file(s)."
rpm -ivh --nodeps "$SCRIPT_DIR"/*.rpm

echo "Configuring Docker runtime..."
nvidia-ctk runtime configure --runtime=docker

echo "Restarting Docker..."
systemctl restart docker

echo "Verifying GPU access in Docker..."
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

echo "Done."
""")
print(f"  install_offline.sh written.")

print()
print("Files created:")
for f in sorted(os.listdir(dest_dir)):
    size = os.path.getsize(os.path.join(dest_dir, f))
    print(f"  {f:<55} {size/1024:.0f} KB")

print()
print("IMPORTANT: nvidia-container-toolkit RPMs require fetching from RHEL.")
print("  Option A: Run fetch_toolkit_rpms_on_rhel.sh on a RHEL VM with internet.")
print("  Option B: On the airgapped machine add repo file and use:")
print("            sudo dnf install --disablerepo='*' --enablerepo=nvidia \\")
print("              nvidia-container-toolkit  (if partial internet allowed)")
