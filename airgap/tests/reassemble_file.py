"""
Reassemble split chunks back into the original file.
Searches disc1, disc2, disc3 folders for chunks named <basename>.001, .002 ...

Usage: reassemble_file.py <disc_root> <original_basename> <output_path>
"""
import sys, os

disc_root = sys.argv[1]
basename  = sys.argv[2]   # e.g. gpu-images-combined.tar
out_path  = sys.argv[3]

# Find all chunks across all disc folders
chunks = {}
for disc in sorted(os.listdir(disc_root)):
    disc_path = os.path.join(disc_root, disc)
    if not os.path.isdir(disc_path):
        continue
    for fname in os.listdir(disc_path):
        if fname.startswith(basename + ".") and fname[-3:].isdigit():
            idx = int(fname[-3:])
            chunks[idx] = os.path.join(disc_path, fname)

if not chunks:
    print(f"[ERROR] No chunks found for '{basename}' under {disc_root}")
    sys.exit(1)

total_chunks = max(chunks.keys())
print(f"  Found {len(chunks)} chunk(s) of {total_chunks} expected.")

# Verify all chunks present
for i in range(1, total_chunks + 1):
    if i not in chunks:
        print(f"[ERROR] Missing chunk {i:03d} — check all DVDs are copied.")
        sys.exit(1)

# Reassemble
total_written = 0
print(f"  Reassembling to: {out_path}")
with open(out_path, "wb") as out:
    for i in range(1, total_chunks + 1):
        chunk_path = chunks[i]
        size = os.path.getsize(chunk_path)
        print(f"  Chunk {i}/{total_chunks}: {os.path.basename(chunk_path)} ({size/1024**2:.0f} MB) ...", end="", flush=True)
        with open(chunk_path, "rb") as c:
            while True:
                buf = c.read(4 * 1024 * 1024)
                if not buf:
                    break
                out.write(buf)
                total_written += len(buf)
        print(" OK")

print(f"\n  Total: {total_written/1024**3:.2f} GB written to {out_path}")
