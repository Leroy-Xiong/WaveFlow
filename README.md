# WaveFlow

**Probabilistic ocean-wave and water-level prediction on unstructured meshes**

WaveFlow is a conditional flow-matching model for ensemble prediction of
significant wave height, peak period, mean direction and water level on
unstructured coastal and global meshes. It conditions on atmospheric fields —
ERA5 reanalysis, or an IFS ensemble when a forecast is being emulated.

This repository contains the **two released checkpoints and the code needed to
sample from them**.

- **Paper**: WaveFlow enables high-resolution prediction of ocean waves and water levels for coastal hazard warning
- **Weights** (CC BY 4.0): [10.5281/zenodo.24106732](https://zenodo.org/records/24591387) · [Hugging Face](https://huggingface.co/tikitaka0345/WaveFlow)
- **Code**: [https://github.com/Leroy-Xiong/WaveFlow](https://github.com/Leroy-Xiong/WaveFlow) · **Licence**: MIT (code), CC BY 4.0 (weights/data)

## Released models

| Checkpoint | Domain | Variables | Nodes | Params | Epoch |
|---|---|---|---|---|---|
| `waveflow-hk-4var-ep49.pt` | Hong Kong waters | Hs, Tp, Dm, water level | 7,788 | 45.69 M | 49 |
| `waveflow-global-4var-ep199.pt` | global ocean | Hs, Tp, Dm, water level | 651,652 | 966.04 M | 199 |

All checkpoints are EMA-only: the EMA weights are baked into
`model_state_dict` and the non-EMA buffers are preserved. Load them with
`training.use_ema=false` and `sampling.use_ema_for_sampling=false` (the
`waveflow` CLI does this already).

## Install

```bash
conda env create -f environment.yml
conda activate waveflow
python -m waveflow env                 # packages, CUDA, local weights
python -m waveflow setup --weights     # 183.1 MB + 3,883.3 MB + assets
```

`setup` downloads from Hugging Face, verifies the SHA256 checksums listed in
`waveflow/downloads.yaml`, and extracts the mesh assets (coordinates, patch
clusters, normalization statistics, ocean boundary) into `data/`.

## Sample

```bash
# 10 members on the Hong Kong mesh at one timestamp, ERA5 forcing
python -m waveflow sample --model hk --timestamps 20211013T00 --members 10

# 10 members on the global mesh
python -m waveflow sample --model global --timestamps 20220924T00 --members 10 --steps 50
```

Each member is written to `<out>/samples/sample_<timestamp>_ens<NNN>.npz`:

```python
import numpy as np
d = np.load("runs/hk_20211013T00/samples/sample_20211013T00_ens000.npz")
d["sample"]        # (n_nodes, 4) float32, columns hs, tp, dm, elev (physical units)
d["timestamp"]     # "20211013T00"
d["ensemble_index"]  # 0
```

The atmospheric forcing comes from WeatherBench2 on demand (public, anonymous
access); `--era5-dir` points the loader at your own NetCDF frames instead.
See `INFERENCE.md` for the ensemble-of-forcing mode (`--ens-members`), the
output layout, memory and timing.

## Real-time forecast

```bash
python -m waveflow forecast --init latest --leads 24,48,72 --members 10
python -m waveflow forecast --model global --leads 24,48 --members 10
python -m waveflow forecast --init 20261127T18 --leads 24-120:24 --plot
```

The forcing comes from the **IFS ensemble forecast on ECMWF open data**
(stream `enfo`, 0.25°, four cycles a day, steps 0–144 h every 3 h), not from
reanalysis: this is the operational mode. The command picks the newest
published cycle, fetches the three surface fields for the requested members,
averages them, and runs the Hong Kong checkpoint at every lead. Each lead
writes `<out>/lead_<L>h/samples/*.npz`.

GRIB decoding needs `eccodes` (`conda install -c conda-forge eccodes`, or
`pip install --target .deps eccodes`); everything else is already in
`environment.yml`. See `INFERENCE.md` for cycle timing, the frame layout and
the lead-time semantics.

## Plot

```bash
python -m waveflow plot --run runs/hk_20211013T00          # newest run if omitted
python -m waveflow plot --run runs/hk_20211013T00 --var tp --out tp.png
python -m waveflow sample --model hk --timestamps 20211013T00 --plot
```

Each figure shows one member, the ensemble mean and the ensemble spread on the
model mesh, with Natural Earth coastlines; `--var` picks `hs`, `tp`, `dm` or
`elev`. Figures land in `<run>/figures/`.

## Citing

See `CITATION.cff`. The weights carry their own DOI
([10.5281/zenodo.24106732](https://zenodo.org/records/24591387)); the concept DOI always resolves to the
latest version.
