# Data

This repository redistributes **code, trained weights and model assets**. It
does **not** redistribute the hindcast targets, and it needs no private data to
sample.

## 1. Model assets (`assets.tar.gz`)

`python -m waveflow setup --weights` downloads 22.1 MB and extracts it at
the repository root. It restores everything the checkpoints need besides the
weights:

| Path | Content |
|---|---|
| `data/hindcast/hk_coordinates.npy` | Hong Kong mesh, 7,788 nodes, `(lat, lon)` |
| `data/hindcast/hk_clusters_512.npz` | 512 patches × 16 nodes, labels + centres |
| `data/hindcast/wave_data_stats.json` | Hong Kong normalization statistics |
| `data/global_hindcast/global_coordinates.npy` | global mesh, 651,652 nodes |
| `data/global_hindcast/global_clusters_2048.npz` | 2,048 patches × 319 points |
| `data/global_hindcast/statistics_4var.json` | global four-variable statistics |
| `data/global_hindcast/statistics.json` | legacy single-variable statistics |
| `data/global_hindcast/ocean_boundary_wkb.npz` | ocean boundary geometry |
| `data/era5/era5_statistics{,_global}.json` | ERA5 normalization statistics |

The normalization values are documented in `NORMALIZATION.md`.

## 2. Atmospheric forcing (public)

| Source | Use | Access |
|---|---|---|
| ERA5 reanalysis, 0.25°, 3 surface variables (`u10`, `v10`, `msl`) | default inference forcing | WeatherBench2 `gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`, anonymous GCS access |
| IFS ENS, 0.25°, 10 members | forecast emulation (`--ens-members`) | WeatherBench2 `gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr`, anonymous GCS access |

The CLI fetches and caches these fields on demand under `data/era5/wb2/` and
`data/ifs/wb2_ens/`. ERA5 is Copernicus data (Copernicus licence, attribution);
WeatherBench2 is distributed under CC BY 4.0.

## 3. Hindcast targets (not redistributed)

The Hong Kong and global wave/water-level hindcasts are third-party products.
They were used to train the released models; **sampling does not read them**.
The CLI writes zero-valued anchor files instead, because the sampler consumes
only the timestamp and the atmosphere.

If you have your own hindcast and want to evaluate the model, the layout the
code expects is:

- Hong Kong: `data/hindcast/processed_npz/wave_<YYYYMMDDTHH>.npz` with keys
  `hs`, `tp`, `dm`, `elev`, each `(7788,)`.
- Global four-variable: `data/global_hindcast/npy_files_daily_4var/<yyyy>/global_hindcast_<yyyymmdd>.npy`,
  shape `(8, 651652, 4)`.

Model outputs are plain `.npz` files (see `INFERENCE.md`), so any evaluation
toolchain can read them without this repository.

