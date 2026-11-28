# Inference

`python -m waveflow sample` runs the released checkpoint with the paper's
sampling recipe: conditional flow matching, 50 ODE steps, `v`-prediction, and
the zero-valued "anchor" files that let sampling run without a hindcast target.

```bash
python -m waveflow sample --model hk --timestamps 20211013T00 --members 10
python -m waveflow sample --model global --timestamps 20220924T00,20220924T06 \
    --members 10 --out runs/global_two_frames
```

| Flag | Meaning |
|---|---|
| `--model {hk,global}` | which checkpoint and mesh to use |
| `--timestamps` | comma-separated valid times, `YYYYMMDDTHH` |
| `--members` | ensemble members (default 10) |
| `--steps` | ODE steps (default 50) |
| `--batch-size` | override `sampling.batch_size` (HK 32, global 2) |
| `--out` | run directory (default `runs/<model>_<first timestamp>`) |
| `--era5-dir` | use your own NetCDF ERA5 frames instead of WeatherBench2 |
| `--nproc` | number of GPUs (default 1) |
| `--dry-run` | print the commands without running them |

## Forcing

By default the ERA5 frames are fetched from **WeatherBench2** (anonymous GCS
access, `gs://weatherbench2/datasets/era5/...`) and cached under
`data/era5/wb2/{hk,global}/`. The region, frame count (8 for Hong Kong, 4 for
the global mesh) and file format are config overrides applied by the CLI, so
the shipped configs can keep pointing at the training layout.

To use your own forcing, pass `--era5-dir <dir>` containing
`<year>/era5_<YYYYMMDDTHH>.nc`, the same layout the fetcher produces.

### Emulating a forecast (IFS ENS members)

The paper's forecast cases force each ensemble member with a different member
of an IFS ensemble instead of reanalysis:

```bash
python -m waveflow sample --model global --timestamps 20220928T12 \
    --members 10 --ens-members 10 --ens-init 20220927T12 --ens-lead 24
```

This sets `sampling.member_forced=true`, `sampling.atm_members=N` and
`sampling.atm_source=ens`, and downloads the member fields from the
WeatherBench2 IFS ENS archive. Without `--ens-members` every member is forced
by the same ERA5 field.

## Real-time forecast forcing

`python -m waveflow forecast` runs the operational path: instead of reanalysis
or the archived IFS ENS in WeatherBench2, the atmosphere comes from the
**IFS ensemble forecast published on ECMWF open data**.

| Item | Value |
|---|---|
| Stream | `enfo` (IFS ENS), 0.25° |
| Cycles | 00/06/12/18 UTC, published ~6.5–7.6 h after the nominal time |
| Steps | 0–144 h every 3 h, 50 perturbed members |
| Fields fetched | `10u`, `10v`, `msl` only, one HTTP Range request per field (~0.75 MB) |
| Forcing used | ensemble mean over `--members` (default 10) |

```bash
python -m waveflow forecast --init latest --leads 24,48,72 --members 10
python -m waveflow forecast --init 20261127T18 --leads 24-120:24 --plot
```

`--init latest` probes `data.ecmwf.int` for the newest cycle whose step-0 index
is online. `--leads` accepts a list (`24,48,72`) or a range (`24-120:24`); each
lead produces its own run directory, `runs/forecast_<init>/lead_<L>h/`.

The model consumes eight 6-hourly frames ending at the valid time, so a lead
needs the files: frames inside the cycle come from that cycle, earlier padding
frames from the most recent 00/12Z cycle. They are written to

```
<out>/forcing/lead_time_<L>h/<year>/ifs_ens_<init>.nc   # 40x40, 3 fields
```

which is the layout `WaveDataset` expects for `atm_source='ens'`. Sampling then
draws `--members` stochastic members from that mean forcing, one per lead.

Two notes:

- GRIB decoding needs `eccodes` (`conda install -c conda-forge eccodes`, or
  `pip install --target .deps eccodes cfgrib`); the command says so if missing.

### `--model global`

The global checkpoint takes ERA5-format global fields instead of the 40×40
window, so the same fetch is written differently:

```
<out>/era5_global/<year>/era5_<valid>.npz     3 fields, 721x1440, + era5_coordinates.npz
<out>/stubs/npy_files_4var/<year>/...         zero-valued wave anchors
```

Four frames 6 h apart ending at the valid time, `data.era5_file_format=npz` and
`data.use_preinterpolated_era5=false`. The stored longitude axis is -180..180,
matching the field ordering that `eccodes` returns: `WaveDataset` only re-sorts
the columns when the stored longitudes exceed 180, and storing 0..360 there
would shift the forcing by 180°.

```bash
python -m waveflow forecast --model global --leads 24,48 --members 10
```

## Output

```
<out>/samples/sample_<timestamp>_ens<NNN>.npz
<out>/stubs/...          zero-valued wave anchors (reusable)
<out>/raw/...            per-lead sample trees
```

Each `.npz` contains:

| Key | Content |
|---|---|
| `sample` | `(n_nodes, 4)` float32, columns `hs, tp, dm, elev` in physical units |
| `timestamp` | valid time string |
| `ensemble_index` | member index |
| `sampling_steps` | ODE steps used |
| `rank` | distributed rank that wrote the file |

Node order matches `data/hindcast/hk_coordinates.npy` (Hong Kong) or
`data/global_hindcast/global_coordinates.npy` (global), both row order
`(lat, lon)`.

## Quick-look plots

```bash
python -m waveflow plot --run runs/global_20220924T00 --var hs
```

Three panels — first member, ensemble mean, ensemble spread — drawn on the
model mesh with cartopy coastlines (PlateCarree for Hong Kong, Robinson for the
global mesh, where latitudes beyond ±87° are clipped). Four variables are
supported: `hs` (turbo), `tp` (cividis), `dm` (twilight), `elev` (RdBu_r,
symmetric about zero). The colour limits use the 99.5th percentile of the
ensemble mean, so an individual outlier cannot flatten the map.

`sample --plot` calls the same code when sampling finishes; `--plot-var`
selects the variable.

## Memory, time and numerics

| Model | Batch | Peak GPU memory | 10 members, 1 timestamp |
|---|---|---|---|
| HK 4-var | 32 | ≈ 6 GB | ≈ 1 min |
| Global 4-var | 2 | ≈ 11 GB | ≈ 5 min (single 48 GB card) |

If you hit an out-of-memory error, lower `--batch-size` first. Sampling is
deterministic per member given the seed, but different cards/driver versions
can produce bitwise-different ensembles; the ensemble statistics are stable.

## Reproducibility caveats

- The forcing is re-fetched from WeatherBench2, which publishes fixed
  reanalysis fields; the archive may add newer years but does not change old
  ones.
- The sampler runs in BF16 autocast where the config enables it; disable AMP in
  the config if you need full float32 output.
