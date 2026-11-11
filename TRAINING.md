# Training and fine-tuning WaveFlow

WaveFlow is trained with conditional flow matching (CFM): the model learns a
velocity field that transports Gaussian noise to the conditional distribution
of the target variables, conditioned on atmospheric fields, mesh coordinates
and optional tide phase. The training loop is the same `main.py` entry point
used for sampling:

```bash
torchrun --nproc_per_node=<N_GPUS> main.py --mode train --config <config.yaml> \
  -o section.key=value ...
```

The three files in `configs/release/` are the exact recipes of the released
checkpoints. They are sampling-ready (EMA baked in); for training set
`training.use_ema` as you prefer and point `data.*` at your own files.

## Hardware

| Model | Sampling | Training (reference run) |
|---|---|---|
| Hong Kong 4-var | 1 × 24 GB | 2 × 24 GB (DDP, BF16) |
| Global Hs | 1 × 80 GB | multi-GPU 80 GB class |
| Global 4-var | 1 × 80 GB | multi-GPU 80 GB class |

The global models use `torch.compile`, BF16 AMP and activation checkpointing
where configured; reduce `training.batch_size` or `dataset.num_workers` on
smaller nodes. DDP is set up by `main.py`; launch with `torchrun`.

## Data contract

### Mesh assets

Every experiment needs three files:

| Config key | Content |
|---|---|
| `data.coord_file_path` | node coordinates, `(N_nodes, 2)` for the HK mesh or the spherical/3-D encoding used by the global mesh |
| `data.cluster_file` | patch clustering NPZ: `labels`, `centers`, `cluster_sizes`, `n_clusters`, `coordinates`, `max_points`, `local_coords_template` (and `local_coords_3d_template` for global) |
| `data.stats_file` | normalization statistics JSON (see below) |

The released `assets.tar.gz` contains the exact files for the three released
meshes. If your mesh matches, reuse them; otherwise you must produce a
compatible clustering (same array names and node ordering as your wave arrays).

### Wave targets

**Hong Kong (`region: hk`, `file_format: npz`)**

```
data_dir/wave_YYYYMMDDTHH.npz
  hs   (7788,)  float32   significant wave height, m
  tp   (7788,)  float32   peak period, s
  dm   (7788,)  float32   mean direction, deg
  elev (7788,)  float32   water level, m
```

One file per hourly timestamp; the file name supplies the timestamp. The
loader splits by year: `<= data.train_max_year` for training, 2019–2020 for
validation and 2021–2022 for test (edit `modules/datasets/wave_dataset.py` if
your years differ).

**Global (`region: global`, `file_format: npy` or `npy_daily`)**

```
data_dir/global_hindcast_YYYYMMDD_HH.npy     # (651652,) or (651652, 4)
data_dir/wave_columns.json                   # {"columns": ["hs","tp","dm","elev"]}
```

For the four-variable model the last axis follows `wave_columns.json`; the
Hs-only model uses a 1-D array. `npy_daily` packs eight 3-hourly slots per day
for faster I/O (see `modules/preprocess/convert_npy_to_daily.py` and
`pack_daily.py`).

### Atmospheric forcing

| Config key | Hong Kong | Global |
|---|---|---|
| `data.era5_dir` | `npz` per timestamp, 8 frames | daily-packed `npy` (or per-timestamp `npy`) |
| `data.era5_file_format` | `npz` | `npy_daily` |
| `data.use_preinterpolated_era5` | false | true (recommended) |
| `data.era5_stats_file` | `era5_statistics.json` | `era5_statistics_global.json` |
| `model.era5_num_frames` | 8 | 4 |
| `sampling.atm_source` | `era5` / `ens` | `era5` / `ens` |

`atm_source` selects `era5` (reanalysis, used for pretraining) or `ens` (IFS
ensemble forecast). ERA5 is available from the Copernicus Climate Data Store
and the WeatherBench2 archive. IFS ENS is available from WeatherBench2
(`gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`) or ECMWF open
data; export per-member files under

```
{ens_dir}/lead_time_<L>h/<YYYY>/ifs_ens_m<MM>_<init>.nc
{ens_dir}/lead_time_<L>h/<YYYY>/ifs_ens_<init>.nc      # ensemble mean
```

and set `sampling.atm_source=ens`, `sampling.ens_dir` and
`sampling.atm_members`.

The ERA5/IFS frames must cover the history window of every target timestamp
(`era5_num_frames` × 6 h). For the global models the pre-interpolated daily
packs (`era5_preinterp_<day>.npy`, shape `(8, N_nodes, 3)` plus
`era5_daily_ts_index.npz`) avoid re-interpolating on the fly; build them with
`modules/preprocess/build_global_mmap_parallel.py`,
`interpolate_global_grid_to_nodes.py` and `pack_daily.py`. ERA5 itself is not
redistributed (see `DATA.md`).

### Normalization statistics

`data.stats_file` is a JSON object with, for each variable, the keys
`<var>_min`, `<var>_max`, `<var>_mean`, `<var>_std` (the HK file additionally
has `hs_log_mean/std` and `tp_log_mean/std` for the log transforms). The
released `data/*/wave_data_stats.json` / `statistics.json` are working
examples; recompute them for your own data. The transform code is in
`modules/datasets/wave_normalize.py`.

## Train from scratch

```bash
torchrun --nproc_per_node=2 main.py --mode train \
  --config configs/release/hk_4var.yaml \
  -o data.data_dir=/path/to/hk_npz \
  -o data.coord_file_path=/path/to/hk_coordinates.npy \
  -o data.stats_file=/path/to/wave_data_stats.json \
  -o data.cluster_file=/path/to/hk_clusters_512.npz \
  -o data.era5_dir=/path/to/era5_npz \
  -o training.epochs=50 \
  -o training.batch_size=500 \
  -o training.use_ema=true \
  -o work_dir.path=output/my_hk_train
```

Checkpoints are written under `work_dir.path/checkpoints/` (the exact prefix
comes from `work_dir.prefix`). Resume with:

```bash
... -o checkpoint.resume_path=output/my_hk_train/checkpoints/checkpoint_epoch_10.pt \
    -o checkpoint.resume_training=true -o checkpoint.load_optimizer=true
```

## Fine-tune a released checkpoint

Use the released weights as initialization, start a fresh optimizer and a
lower learning rate:

```bash
torchrun --nproc_per_node=2 main.py --mode train \
  --config configs/release/hk_4var.yaml \
  -o checkpoint.resume_path=weights/waveflow-hk-4var-ep49.pt \
  -o checkpoint.resume_training=false \
  -o checkpoint.load_optimizer=false \
  -o training.use_ema=true \
  -o training.learning_rate=1e-5 \
  -o data.data_dir=/path/to/your_npz \
  -o data.era5_dir=/path/to/your_era5 \
  -o work_dir.path=output/my_hk_finetune
```

The released checkpoint stores EMA weights in `model_state_dict`; with
`training.use_ema=true` a new EMA is initialized from them. If you prefer to
continue without EMA, set `training.use_ema=false`.

## Custom mesh checklist

1. Node order must be identical across coordinates, wave arrays, clusters and
   the ERA5 node interpolation.
2. Build a clustering NPZ with the same keys as the released files; the local
   patch size (`local_coords_template` shape) must match the model's patch
   size and `max_points`.
3. Set `model.in_channels` to the number of target variables and list them in
   `data.variables`.
4. Recompute the statistics JSON and use the same variable names.
5. Start with the HK base config for a small mesh; for a global mesh use
   `global_hs.yaml`/`global_4var.yaml` and the spherical/3-D encodings.
6. Check a forward pass on one batch before launching a long run (`--mode
   sample` on a tiny timestamp set is a quick smoke test).

## Monitoring and outputs

- Logs: console + `work_dir.path/logs/`.
- Checkpoints: `work_dir.path/checkpoints/checkpoint_epoch_<N>.pt` at
  `training.checkpoint_interval`; EMA shadow keys are saved by the trainer.
- Sampling after training uses the same config with `--mode sample` and
  `-o checkpoint.resume_path=...`.

## Common pitfalls

- **Year split**: HK training with `train_max_year` keeps 2019–2020 as
  validation and 2021–2022 as test; a custom year range needs a code/config
  change.
- **ERA5 coverage**: a missing frame silently reduces the file list; the
  loader raises “No files with corresponding atmospheric data” if everything
  is filtered out.
- **Statistics mismatch**: wrong `hs_log_mean/std` or variable order produces
  finite but badly scaled samples; always check a reconstruction against the
  raw data.
- **EMA**: sampling configs must set `training.use_ema=false` and
  `sampling.use_ema_for_sampling=false` when loading the released
  EMA-baked checkpoints; otherwise the sampler looks for a shadow that is not
  there.
