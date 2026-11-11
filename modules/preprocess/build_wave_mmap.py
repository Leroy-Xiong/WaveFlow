"""Build monolithic mmap cache from per-timestamp global wave NPZ files.
Resumes from last written row if mmap file already exists.
"""
import numpy as np
import os
import glob
import sys

def main():
    data_dir = './data/global_hindcast/npz_files'
    all_files = sorted(glob.glob(os.path.join(data_dir, '**/global_hindcast_*.npz'), recursive=True))
    print(f"Total NPZ files: {len(all_files)}", flush=True)

    # Filter: training set is 1959-2020
    def extract_ts(filepath):
        fname = os.path.basename(filepath)
        parts = fname.replace('global_hindcast_', '').replace('.npz', '').split('_')
        return parts[0] + 'T' + parts[1]

    train_files = []
    for f in all_files:
        ts = extract_ts(f)
        year = int(ts[:4])
        if 1959 <= year <= 2020:
            train_files.append((f, ts))

    n_ts = len(train_files)
    print(f"Training timestamps: {n_ts}", flush=True)
    print(f"First: {train_files[0][1]}, Last: {train_files[-1][1]}", flush=True)

    # Save timestamp index
    ts_list = [ts for _, ts in train_files]
    idx_path = './data/global_hindcast/wave_mmap_ts_index.npz'
    np.savez(idx_path, timestamps=np.array(ts_list), indices=np.arange(n_ts))
    print(f"Saved timestamp index to {idx_path}", flush=True)

    # Build or resume monolithic mmap
    cache_path = './data/global_hindcast/global_wave_all_frames.npy'
    n_nodes = 651652
    n_vars = 3
    shape = (n_ts, n_nodes, n_vars)
    size_gb = np.prod(shape) * 4 / 1024**3
    print(f"Cache shape: {shape} = {size_gb:.1f} GB at {cache_path}", flush=True)

    # Resume from existing file or create new
    start_idx = 0
    if os.path.exists(cache_path):
        existing_mm = np.load(cache_path, mmap_mode='r')
        # Find last valid (non-zero) row via binary search from end
        lo, hi = 0, n_ts - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if (existing_mm[mid] == 0).all():
                hi = mid
            else:
                lo = mid + 1
        start_idx = lo if (existing_mm[lo] == 0).all() else lo + 1
        print(f"Resuming from row {start_idx}/{n_ts} ({100*start_idx/n_ts:.1f}%)", flush=True)
        del existing_mm

    mm = np.lib.format.open_memmap(cache_path, mode='r+', dtype=np.float32, shape=shape)

    errors = 0
    for i in range(start_idx, n_ts):
        f, ts = train_files[i]
        if i % 2000 == 0:
            mm.flush()
            print(f"  {i}/{n_ts} ({100*i/n_ts:.1f}%), errors={errors}, ts={ts}", flush=True)
        try:
            d = np.load(f, allow_pickle=True)
            mm[i, :, 0] = d['hs'].astype(np.float32)
            mm[i, :, 1] = d['mwd'].astype(np.float32)
            mm[i, :, 2] = d['ssh'].astype(np.float32)
            d.close()
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  SKIP {f}: {e}", flush=True)
            if errors > 100:
                print(f"  Too many errors ({errors}), aborting", flush=True)
                sys.exit(1)

    mm.flush()
    print(f"Done! {n_ts} rows, {errors} skipped. File: {os.path.getsize(cache_path)/1024**3:.1f} GB", flush=True)

    # Verify
    print(f"Verify: first row hs=[{mm[0,:,0].min():.2f}, {mm[0,:,0].max():.2f}]", flush=True)
    print(f"Verify: last row hs=[{mm[-1,:,0].min():.2f}, {mm[-1,:,0].max():.2f}]", flush=True)

if __name__ == '__main__':
    main()
