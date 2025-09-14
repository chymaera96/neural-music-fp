import os
import numpy as np
from tqdm import tqdm
from glob import glob
import argparse

parser = argparse.ArgumentParser(
    description="Create a large memmap array from many small (n,D) .npy files.")
parser.add_argument("--root", type=str, required=True,
                    help="Root directory containing .npy files (recursively).")
parser.add_argument("--db_name", type=str, default="dummy_db.mm",
                    help="Output memmap file name.")

args = parser.parse_args()

# collect .npy files recursively
all_files = glob(os.path.join(args.root, "**", "*.npy"), recursive=True)
print(f"Found {len(all_files)} files.")

# determine D and dtype
sample = np.load(all_files[0], mmap_mode="r")
n, D = sample.shape
dtype = sample.dtype

# compute total N
N = sum(np.load(f, mmap_mode="r").shape[0] for f in all_files)
print(f"Final memmap shape: ({N}, {D}), dtype={dtype}")

# create memmap
out_path = os.path.join(args.root, args.db_name)
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
base, _ = os.path.splitext(args.db_name)
shape_path = os.path.join(args.root, f"{base}_shape.npy")
np.save(shape_path, np.array([N, D], dtype=np.int64))

print(f"Saved memmap to {out_path} with shape ({N}, {D})")
print(f"Shape written to {shape_path}")
