# WaveFlow

**Probabilistic ocean-wave and water-level prediction on unstructured meshes**

WaveFlow is a conditional flow-matching model for ensemble prediction of
significant wave height, peak period, mean direction and water level on
unstructured coastal/global meshes. It conditions on atmospheric fields
(ERA5 reanalysis, IFS ensemble forecasts, or AI weather forecasts such as
GenCast) plus optional tide phase, and generates calibrated ensembles at every
mesh node.

This repository contains the model, the three final checkpoints, the exact
training/sampling configs, and the scripts and source data needed to rebuild
the paper figures.

- **Paper**: WaveFlow enables high-resolution prediction of ocean waves and water levels for coastal hazard warning
- **Weights and figure source data** (CC BY 4.0): [Hugging Face](https://huggingface.co/tikitaka0345/WaveFlow)
- **Code**: [https://github.com/Leroy-Xiong/WaveFlow](https://github.com/Leroy-Xiong/WaveFlow) · **Licence**: MIT (code), CC BY 4.0 (weights/data)

## Released models

| Checkpoint | Domain | Variables | Nodes | Params | Epoch |
|---|---|---|---|---|---|
| `waveflow-hk-4var-ep49.pt` | Hong Kong waters | Hs, Tp, Dm, water level | 7,788 | 45.7 M | 49 |
| `waveflow-global-hs-ep99.pt` | global ocean | Hs | 651,652 | 465 M | 99 |
| `waveflow-global-4var-ep199.pt` | global ocean | Hs, Tp, Dm, water level | 651,652 | 966 M | 199 |

All checkpoints are EMA-only (the EMA weights are baked into
`model_state_dict`; the other buffers are preserved). Load them with
`training.use_ema=false` and `sampling.use_ema_for_sampling=false`.

## Quick start

### 1. Install

```bash
git clone https://github.com/Leroy-Xiong/WaveFlow waveflow
cd waveflow
conda env create -f environment.yml
conda activate waveflow
```

A pip-only recipe is in `requirements-pip.txt`; an NVIDIA GPU with CUDA is
required for sampling and training (CPU rendering works for the figure
panels only).

### 2. Get the weights

```bash
python reproduce.py setup --weights       # 5.9 GB, SHA256 checked
```

or download the three `.pt` files plus `assets.tar.gz` directly from
[Hugging Face](https://huggingface.co/tikitaka0345/WaveFlow); `assets.tar.gz` restores the mesh
coordinates, clusters, statistics and ocean boundary under `data/`.

### 3. Run an ensemble

The tested entry points for the paper cases:

```bash
python reproduce.py infer fig3f    # global Hs, 1 timestamp x 10 members
python reproduce.py infer ed5      # Hong Kong 4-var, 2 timestamps x 10
python reproduce.py infer ed6      # global 4-var, 2 timestamps x 10
python reproduce.py infer fig6     # Hong Kong 4-var, 49 h x 100 + overtopping
```

For your own inputs, call the sampler directly. The pattern for the Hong Kong
model is:

```bash
torchrun --nproc_per_node=1 main.py --mode sample \
  --config configs/release/hk_4var.yaml \
  -o checkpoint.resume_path=weights/waveflow-hk-4var-ep49.pt \
  -o checkpoint.resume_training=false \
  -o checkpoint.load_optimizer=false \
  -o training.use_ema=false \
  -o sampling.use_ema_for_sampling=false \
  -o data.data_dir=/path/to/your/processed_npz \
  -o sampling.atm_source=era5 \
  -o sampling.target_timestamps=20220701T20 \
  -o sampling.ensemble_size=10 \
  -o work_dir.path=output/my_hk_run
```

For AI forecast forcing, add
`-o sampling.atm_source=gencast -o sampling.gencast_dir=/path/to/gencast_global`
(and `-o sampling.atm_members=<N>` for per-member files). The loader expects
`{gencast_dir}/lead_time_<L>h/<YYYY>/gencast_m<MM>_<init>.nc`; the export path
and other AI forecast layouts are documented in `TRAINING.md`.

Samples are written to
`output/my_hk_run/samples/sample_<timestamp>_ens<NNN>.npz` under the key
`sample`, shape `(N_nodes, N_variables)` for the four-variable models and
`(N_nodes,)` for the global Hs model. The matching ERA5 frames must exist in
the configured `era5_dir`; see `WEIGHTS.md` and `DATA.md`.

Hardware: the Hong Kong model fits a 24 GB GPU; the global models need about
80 GB (BF16, batch 1–16). CPU sampling is intentionally refused because it does
not reproduce the GPU result; CPU is enough for the figure-rendering path.

### 4. Train or fine-tune on your own data

Training uses the same `main.py` entry point:

```bash
torchrun --nproc_per_node=2 main.py --mode train \
  --config configs/release/hk_4var.yaml \
  -o data.data_dir=/path/to/your/processed_npz \
  -o data.coord_file_path=/path/to/your/coordinates.npy \
  -o data.stats_file=/path/to/your/statistics.json \
  -o data.cluster_file=/path/to/your/clusters.npz \
  -o data.era5_dir=/path/to/your/era5_npz \
  -o training.epochs=50 \
  -o training.use_ema=true \
  -o work_dir.path=output/my_hk_train
```

The data contract, file naming, ERA5 layouts, custom-mesh notes and
fine-tuning recipes are in **`TRAINING.md`**. In short:

- **Hong Kong format**: one `wave_YYYYMMDDTHH.npz` per timestamp with the
  arrays `hs`, `tp`, `dm`, `elev` of shape `(N_nodes,)`.
- **Global format**: one `global_hindcast_YYYYMMDD_HH.npy` per timestamp, shape
  `(N_nodes,)` for Hs or `(N_nodes, 4)` with a `wave_columns.json`; the release
  configs use daily-packed pre-interpolated ERA5 for speed.
- You also need mesh coordinates, patch clusters and normalization statistics;
  the released `assets.tar.gz` contains the ones used by these checkpoints.

The third-party hindcast, ERA5/IFS reanalysis and AI forecast fields are **not**
redistributed;
bring your own copy (see `DATA.md`).

## Reproduce the paper figures

Figure reproduction is one feature of this release, not its only purpose.
The exact samples, derived statistics and reference crops are shipped, so the
panels can be rebuilt on CPU without re-running inference:

```bash
python reproduce.py setup --download   # weights + figure data (6.9 GB)
python reproduce.py list
python reproduce.py all                # panels that need no hindcast
python reproduce.py verify             # pixel comparison against reference/
```

Outputs land in `reproduced/`. The full panel map, hindcast requirements and
GPU re-inference recipes are in `REPRODUCE.md`.

## Repository layout

```
main.py              training / sampling entry point
modules/             model, datasets, training loop, plotting
configs/release/     exact configs of the three released checkpoints
reproduce/           figure-reproduction CLI and panel renderers
reference/           paper figures + exact panel crops used for verification
scripts/             paper figure scripts and release overrides
weights/             downloaded checkpoints (gitignored)
data/, output/       downloaded assets/figure data (gitignored)
```

## Data and licensing

- Code: MIT (`LICENSE`).
- Weights: CC BY 4.0 (see `LICENSES.md`).
- Figure source data: CC BY 4.0; third-party assets retain their own terms
  (`THIRD_PARTY.md`).
- Hindcast targets: third-party, not redistributed (`DATA.md`).
- ERA5, IFS/WeatherBench2, GenCast/GraphCast, CMEMS, HKO and LandsD sources are
  listed in `DATA.md` and `THIRD_PARTY.md`.

## Citation

If you use WaveFlow, please cite the paper and the dataset records in
`CITATION.cff`. Questions: TODO <TODO>.
