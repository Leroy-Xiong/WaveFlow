# Released weights

Three final checkpoints are released under **CC BY 4.0**. The weight files are
hosted on [Hugging Face](https://huggingface.co/tikitaka0345/WaveFlow); the [Zenodo
record](https://zenodo.org/records/24127153) provides the DOI and the SHA256 checksums.

| Name | Model | Params | Size | Epoch | Config |
|---|---|---|---|---|---|
| `waveflow-hk-4var-ep49.pt` | Hong Kong, four variables | 45.69 M | 183.1 MB | 49 | `configs/release/hk_4var.yaml` |
| `waveflow-global-hs-ep99.pt` | global mesh, Hs only | 465.46 M | 1,881.0 MB | 99 | `configs/release/global_hs.yaml` |
| `waveflow-global-4var-ep199.pt` | global mesh, four variables | 966.04 M | 3,883.3 MB | 199 | `configs/release/global_4var.yaml` |

SHA256 and exact source-run metadata: `weights/MANIFEST.json` (after
`python reproduce.py setup --weights`).

## Format

Each file is a PyTorch checkpoint with:

```python
{
  "model_state_dict": {...},          # EMA weights baked in, buffers preserved
  "ema_state_dict": {"shadow": {...}, "decay": ..., "step": ...,
                     "update_after_step": ...},
  "epoch": int,
  "release": {"version": ..., "source_run": ..., "source_checkpoint": ...,
              "exported_utc": ...}
}
```

The EMA shadow was merged into `model_state_dict` (parameters overwritten,
non-EMA buffers kept from the training checkpoint) so sampling can run with
`training.use_ema=false` and `sampling.use_ema_for_sampling=false`, which halves
the GPU memory of the giant model. Loading with the EMA machinery on would
apply the same shadow and produce identical weights.

Key prefixes from `torch.compile` (`_orig_mod.`) were removed. Build the model
with `model.compile=false` (the sampling default) as in the released configs.

## Loading

```bash
torchrun --nproc_per_node=1 main.py \
  --config configs/release/hk_4var.yaml --mode sample \
  -o checkpoint.resume_path=weights/waveflow-hk-4var-ep49.pt \
  -o training.use_ema=false \
  -o sampling.use_ema_for_sampling=false \
  -o data.data_dir=<wave-anchor-dir> \
  -o sampling.ensemble_size=10 \
  -o work_dir.path=output/repro/hk_case
```

`reproduce.py infer ...` builds exactly this command (including the zero-valued
wave anchors, see `reproduce/make_case_stubs.py`).

## Source runs

| Model | Source run | Source checkpoint |
|---|---|---|
| HK 4-var | `output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/base_v1_20260821_025917` | `checkpoint_epoch_49.pt` |
| Global Hs | `output/results/global/new_data_hs/patch2048_4frames_concat_wucos_ema999gc_xpred_huge/train_20260724_124120` | `checkpoint_epoch_99.pt` |
| Global 4-var | `output/results/global/new_data_allvar/patch2048_4frames_concat_xpred_giant_tod_ext/train_npy_bs24_pf2_asym0_20260922_142002` | `checkpoint_epoch_199_ema.pt` |

The training data (hindcast) is not redistributed, so retraining is not
reproducible without provider access; the code, configurations, EMA recipe and
training entry point are included.

## Expected scores

| Quantity | Value |
|---|---|
| HK Hs RMSE (2021–2022 test, 10-member mean) | 0.055 m |
| Global Hs RMSE (5,840 test snapshots) | 0.204 m |
| HK fair CRPS after bc3 | 0.0279 m |
| Global fair CRPS (10 % initialisations) | 0.114–0.126 m |

These are ensemble statistics; a re-run with the released weights reproduces
them within sampling noise.

## Reproducibility note

The model is a conditional flow-matching generator: every member starts from
independent Gaussian noise. Re-running inference therefore produces different
individual members (and slightly different ensemble means) from the paper run.
For the inference tier, expect:

- same shapes, finite values and physical ranges;
- ensemble-mean differences of a few percent (documented tolerances in
  `REPRODUCE.md` §6);
- identical results only if the same GPU count/batch layout and RNG stream are
  reproduced, which is not guaranteed across hardware.

## GPU requirements

| Model | Nodes | Recommended GPU | Notes |
|---|---|---|---|
| HK 4-var | 7,788 | ≥ 24 GB | 100-member Fig. 6 run fits 24–48 GB with batch 32 |
| Global Hs | 651,652 | ≈ 80 GB | 465 M params, batch 16 |
| Global 4-var | 651,652 | ≈ 80 GB | 966 M params, batch 1–2 |
