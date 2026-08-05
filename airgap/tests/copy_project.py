"""
Copy project code to DVD disc3, excluding large/irrelevant folders.
Usage: copy_project.py <source_root> <dest_dir>
"""
import sys, os, shutil

src  = os.path.abspath(sys.argv[1])
dest = os.path.abspath(sys.argv[2])

# Folders to skip entirely
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".kiro",
    "airgap",            # the airgap packages themselves
    "benchmark_results", # large result files not needed
    "spark",             # vendored Spark distribution(s) under cluster/native/spark
                          # (~1 GB) — shipped separately as native/spark/*.tgz, or
                          # not needed at all for a Docker-only airgap target
    "java",              # extracted portable JRE under cluster/native/java —
                          # shipped separately as native/java/*.zip if needed
}

# File extensions to skip
EXCLUDE_EXT = {".tar", ".tgz", ".gz", ".zip", ".tar.gz"}

copied = 0
skipped = 0

for root, dirs, files in os.walk(src):
    # Prune excluded dirs in-place
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

    rel_root = os.path.relpath(root, src)
    dest_root = os.path.join(dest, rel_root)
    os.makedirs(dest_root, exist_ok=True)

    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in EXCLUDE_EXT:
            skipped += 1
            continue
        src_file  = os.path.join(root, fname)
        dest_file = os.path.join(dest_root, fname)
        shutil.copy2(src_file, dest_file)
        copied += 1

print(f"  Copied {copied} files, skipped {skipped} large files.")
