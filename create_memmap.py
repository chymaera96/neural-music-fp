import os
import numpy as np
from tqdm import tqdm
from glob import glob
import argparse
import json

parser = argparse.ArgumentParser(
    description="Create a large memmap array from many small (n,D) .npy files.")
parser.add_argument("--root", type=str,
                    help="Root directory containing .npy files (recursively).")
parser.add_argument("--file_list", type=str,
                    help="JSON file containing a list of .npy file paths.")
parser.add_argument("--db_name", type=str, default="dummy_db.mm",
                    help="Output memmap file name.")

args = parser.parse_args()

# -------------------------
# Collect files
# -------------------------
if args.file_list:
    with open(args.file_list, "r") as f:
        all_files = json.load(f)
    print(f"Loaded {len(all_files)} files from {args.file_list}")
elif args.root:
    all_files = glob(os.path.join(args.root, "**", "*.npy"), recursive=True)
    all_files = [f for f in all_files if not f.endswith("_shape.npy")]
    print(f"Found {len(all_files)} files in {args.root}")
else:
    raise ValueError("You must specify either --root or --file_list")

# -------------------------
# Determine shape & dtype
# -------------------------
sample = np.load(all_files[0], mmap_mode="r")
n, D = sample.shape
dtype = sample.dtype

# compute total N
N = sum(np.load(f, mmap_mode="r").shape[0] for f in all_files)
print(f"Final memmap shape: ({N}, {D}), dtype={dtype}")

# -------------------------
# Create memmap
# -------------------------
out_path = os.path.join(args.root if args.root else ".", args.db_name)
mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=(N, D))

# fill memmap sequentially
offset = 0
for path in tqdm(all_files, desc="Merging"):
    arr = np.load(path)
    n = arr.shape[0]
    mm[offset:offset+n] = arr
    offset += n

mm.flush()

# save shape as a separate .npy file
base, _ = os.path.splitext(out_path)
shape_path = f"{base}_shape.npy"
np.save(shape_path, np.array([N, D], dtype=np.int64))

print(f"Saved memmap to {out_path} with shape ({N}, {D})")
print(f"Shape written to {shape_path}")
