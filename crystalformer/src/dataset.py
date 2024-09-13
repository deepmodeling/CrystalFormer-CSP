import os
import lmdb
import pickle
import numpy as np
from multiprocessing import Pool
from crystalformer.src.utils import GLXYZAW_from_file
import warnings
warnings.filterwarnings("ignore")


def csv_to_lmdb(csv_file, lmdb_file, num_workers=40):
    if os.path.exists(lmdb_file):
        os.remove(lmdb_file)
        print(f"Removed existing {lmdb_file}")

    values = GLXYZAW_from_file(csv_file, atom_types=119, wyck_types=28, n_max=21, num_workers=num_workers)
    keys = np.arange(len(values[0]))

    env = lmdb.open(
        lmdb_file,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    with Pool(num_workers) as pool:
        with env.begin(write=True) as txn:
            for key, value in zip(keys, values):
                txn.put(str(key).encode("utf-8"), pickle.dumps(value))

    print(f"Successfully converted {csv_file} to {lmdb_file}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    num_workers = int(sys.argv[2])

    for i in ["test", "val", "train"]:
        csv_to_lmdb(
            os.path.join(path, f"{i}.csv"), 
            os.path.join(path, f"{i}.lmdb"),
            num_workers=num_workers
        )
