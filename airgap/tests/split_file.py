"""
Split a large file into fixed-size chunks for DVD burning.
Usage: split_file.py <source_file> <output_dir> <chunk_size_bytes>

Chunks are named: <basename>.001, .002, .003 ...
"""
import sys, os

src        = sys.argv[1]
out_dir    = sys.argv[2]
chunk_size = int(sys.argv[3])

basename = os.path.basename(src)
total    = os.path.getsize(src)
chunks   = (total + chunk_size - 1) // chunk_size

print(f"  Source : {src}")
print(f"  Size   : {total/1024**3:.2f} GB")
print(f"  Chunks : {chunks} x {chunk_size/1024**3:.1f} GB each")
print()

with open(src, "rb") as f:
    for i in range(1, chunks + 1):
        chunk_path = os.path.join(out_dir, f"{basename}.{i:03d}")
        written = 0
        print(f"  Writing chunk {i}/{chunks}: {os.path.basename(chunk_path)} ...", end="", flush=True)
        with open(chunk_path, "wb") as out:
            while written < chunk_size:
                buf = f.read(min(4 * 1024 * 1024, chunk_size - written))  # 4 MB reads
                if not buf:
                    break
                out.write(buf)
                written += len(buf)
        size_mb = written / 1024**2
        print(f" {size_mb:.0f} MB  OK")

print(f"\n  All {chunks} chunks written to: {out_dir}")
