import os
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
    description="Create a large memmap array from many small .npy files.")
parser.add_argument(
    "--root",
    type=str,
    required=True,
    help="Root directory containing subfolders with .npy files.")
parser.add_argument(
    "--db_name",
    type=str,
    default="dummy_db.mm")

# root directory containing subfolders 000, 001, ...
args = parser.parse_args()
root = args.root

# collect all .npy files with full paths
all_files = []
for sub in sorted(os.listdir(root)):
    subdir = os.path.join(root, sub)
    if not os.path.isdir(subdir):
        continue
    for f in sorted(os.listdir(subdir)):
        if f.endswith(".npy"):
            all_files.append(os.path.join(subdir, f))

print(f"Found {len(all_files)} files.")

# read one file to get embedding dimension
sample = np.load(all_files[0])
dim = sample.shape[0] if sample.ndim == 1 else sample.shape
print("Embedding shape per file:", dim)

# total number of embeddings
n = len(all_files)

# path for the big memmap
out_path = os.path.join(root, args.db_name)

# create memmap array
mm = np.memmap(out_path, dtype=sample.dtype, mode="w+", shape=(n,) + sample.shape)

# sequentially load and write
for i, path in enumerate(tqdm(all_files, desc="Merging")):
    mm[i] = np.load(path)

# flush changes to disk
mm.flush()
print(f"Saved memmap with shape {mm.shape} at {out_path}")
