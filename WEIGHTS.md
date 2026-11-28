# Released weights

Two final checkpoints are released under **CC BY 4.0**. The files are hosted on
[Hugging Face](https://huggingface.co/tikitaka0345/WaveFlow); the [Zenodo record](https://zenodo.org/records/24591387)
carries the DOI and the SHA256 checksums.

| Name | Model | Params | Size | Epoch | Config |
|---|---|---|---|---|---|
| `waveflow-hk-4var-ep49.pt` | Hong Kong, four variables | 45.69 M | 183.1 MB | 49 | `configs/release/hk_4var.yaml` |
| `waveflow-global-4var-ep199.pt` | global mesh, four variables | 966.04 M | 3,883.3 MB | 199 | `configs/release/global_4var.yaml` |

```bash
python -m waveflow setup --weights      # ~4.1 GB, SHA256 verified
```

## Format

Each file is a PyTorch checkpoint with:

```python
{
  "model_state_dict": {...},   # EMA weights, ready to sample
  "epoch": 49,                 # or 199
  "config": {...},             # the run configuration as recorded at export
}
```

`model_state_dict` also carries the mesh buffers (`patch_centroids`,
`local_coords_template`, `mask_template`, `cluster_labels`), so a checkpoint is
self-describing once `assets.tar.gz` has restored `data/`.

## Loading

The `waveflow` CLI supplies every override that the configs need. To drive
`main.py` yourself:

```bash
python main.py --config configs/release/global_4var.yaml --mode sample \
  -o checkpoint.resume_path=weights/waveflow-global-4var-ep199.pt \
  -o training.use_ema=false \
  -o sampling.use_ema_for_sampling=false \
  -o data.data_dir=data/global_hindcast/case_stubs/npy_files_4var \
  -o sampling.ensemble_size=10 \
  -o sampling.target_timestamps=20220924T00 \
  -o work_dir.path=runs/global_20220924T00
```

## GPU requirements

| Model | Nodes | Sampling memory | Notes |
|---|---|---|---|
| HK 4-var | 7,788 | ≈ 6 GB | `batch_size=32` for 10 members |
| Global 4-var | 651,652 | ≈ 11 GB | `batch_size=2`; 50 ODE steps |

Sampling needs no optimizer state, so a 16–24 GB card reproduces every released
example (measured on an RTX 5880 Ada). The ≈ 80 GB figure sometimes quoted for
this mesh is the training budget.
