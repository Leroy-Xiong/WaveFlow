# Reproducing the figures

This document is the step-by-step companion to `README.md`. All commands are
run from the repository root with the `waveflow` environment active.

## 0. One-time setup

```bash
conda env create -f environment.yml
conda activate waveflow
python reproduce.py setup --weights
```

`setup --weights` checks the environment, downloads the released checkpoints
and model assets from Hugging Face, verifies their SHA256, and reports which
hindcast reference frames are present. Weights land in `weights/`; the asset
archive restores `data/` mesh assets.

The figure-rendering commands assume the data tree used by the paper scripts
is present at the repository root (`data/` and `output/`). `setup` does not
download it.

### Cartopy coastlines

The maps use Natural Earth (auto-downloaded by cartopy) and, for Fig. 2a,
GSHHS full-resolution polygons (also auto-downloaded on first use, ~160 MB).
To pre-fetch:

```bash
python - <<'PY'
import cartopy.io.shapereader as shpreader
shpreader.natural_earth(resolution="10m", category="physical", name="land")
shpreader.gshhs(scale="f", level=1)
PY
```

## 1. Render tier (CPU, deterministic)

`reproduce.py` writes each output to `reproduced/`. Rendering uses the exact
samples and statistics behind the paper figures, so the result is a pixel-level
reproduction (see §6 for tolerances).

| Command | Outputs | Data used | Runtime |
|---|---|---|---|
| `python reproduce.py fig2a` | `reproduced/fig2a.png` | mesh coordinates, one ERA5 frame, static AI ocean mask, GSHHS | ~30 s |
| `python reproduce.py fig3def` | `fig3d.png`, `fig3e.png`, `fig3f.png` | HK/global coordinates + clusters, 10 global samples (2022-09-24 00Z) | ~30 s |
| `python reproduce.py fig6` | `fig6.png` | Ragasa overtopping products (analysis + forecast overlay), 100 peak-hour samples, node cache, coastline/orthophoto/DTM assets | 2–5 min |
| `python reproduce.py ed3` | `ed3.png` | six pairwise 2D histograms, HK + global | ~20 s |
| `python reproduce.py fig5ghi` | `fig5g.png`, `fig5h.png`, `fig5i.png` | shipped case samples + **2 hindcast frames** | ~30 s |
| `python reproduce.py ed5` | `ed5.png` | shipped 2×10 HK samples + **2 hindcast frames** | ~30 s |
| `python reproduce.py ed6` | `ed6.png` | shipped 2×10 global four-variable samples + **2 daily hindcast packs** | 1–5 min |

Run everything that does not need a hindcast frame:

```bash
python reproduce.py all
```

Run everything and fail if a reference-bearing panel cannot be built:

```bash
python reproduce.py all --strict
```

### Panels that need a hindcast reference frame

The hindcast is a third-party product and is not redistributed. The release
needs only these small slices:

| Panel | File | Shape / key |
|---|---|---|
| Fig. 5g | `data/hindcast/processed_npz/wave_20211013T00.npz` | `hs` (7788,) |
| Fig. 5h,i | `data/global_hindcast/npz_files/2022/global_hindcast_20220928_12.npz` | `hs` (651652,) |
| ED 5 | `wave_20220701T20.npz`, `wave_20210125T17.npz` | hs/tp/dm/elev (7788,) |
| ED 6 | `npy_files_daily_4var/2022/global_hindcast_{20220924,20220115}.npy` | (8, 651652, 4) |

Check them with:

```bash
python reproduce.py check-hindcast
```

Point the tools at your copy without moving files:

```bash
export WF_GT_HK=/path/to/hindcast/processed_npz
export WF_GT_GLOBAL=/path/to/global_hindcast/npz_files
export WF_GT_GLOBAL_4VAR=/path/to/global_hindcast/npy_files_daily_4var
python reproduce.py fig5ghi
python reproduce.py ed5
python reproduce.py ed6
```

The same values can be passed as `--gt-hk`, `--gt-global`,
`--gt-global-4var` to any panel command.

## 2. Inference tier (GPU)

`reproduce.py infer <panel>` writes zero-valued wave anchors for the requested
timestamps (the sampler only consumes the timestamp and the atmosphere), runs
`main.py --mode sample` with the released weights, and then re-renders the
panel from the new samples.

```bash
python reproduce.py setup --check-env --infer     # verifies torch + CUDA
python reproduce.py infer fig3f                   # global Hs, 1 timestamp x 10
python reproduce.py infer ed5                     # HK, 2 timestamps x 10
python reproduce.py infer ed6                     # global 4-var, 2 x 10
python reproduce.py infer fig6 --nproc 2          # HK, 49 h x 100 members
```

Approximate cost (one 80 GB GPU unless noted):

| Panel | Model | Members × timestamps | Time |
|---|---|---|---|
| Fig. 3f | global Hs | 10 × 1 | 5–15 min |
| ED 5 | HK 4-var | 10 × 2 | < 5 min |
| ED 6 | global 4-var | 10 × 2 | 30–90 min |
| Fig. 6 | HK 4-var | 100 × 49 | 10–20 GPU·h |

For Fig. 6 the tool also re-runs the EurOtop overtopping computation on the
freshly sampled 100-member analysis and points panels b/d at the new `q`
product (`WF_FIG6_Q_NPZ`). Panel c (M=10 vs M=100 exposure) keeps the shipped
aggregate summary, and the IFS-ENS forecast overlay in panels d/e keeps the
shipped forecast samples + `ragasa_q_forecast_init20250922T12_m50x2.npz`; the
overlay is skipped automatically if those files are absent. Pass
`--which-summary` only if you also re-run the M=10 sampling (240 hourly times,
10 members) and want the summary rebuilt.
`--no-render` stops after sampling; `--render-only` re-renders from an existing
`--out` directory.

### Fig. 5g–i from weights (manual)

The forecast panels use per-member IFS ENS forcing. The release ships the
samples used in the paper; to regenerate them you need the corresponding
forecast forcing (public):

1. Fetch the 10 IFS ENS members for the required initialisations/leads from
   WeatherBench2 (2021–2022 archive) into the layout expected by
   `sampling.atm_source=ens` (`data/ifs/ens_members_hk` for Hong Kong,
   `data/ifs/ens_members` for the global mesh).
2. Run `main.py --mode sample` with `sampling.member_forced=true`,
   `sampling.ens_dir=...`, `sampling.ens_init=<init>`,
   `sampling.ens_lead_time=<lead>`, `sampling.ensemble_size=10`, and the
   timestamps in `panels.yaml` (`20211013T00` for Kompasu, `20220928T12` for
   Ian).
3. Apply the released calibration JSON (`bc2`) if you need the bias-corrected
   maps exactly as in the paper (`scripts/uq/uq_affine_calib.py`).
4. Point the renderer at the new directories:
   `WF_HK_RUN=...` (Kompasu) and `WF_GLOBAL_HS_RUN=...` (Ian), then
   `python reproduce.py fig5ghi`.

## 3. Recomputing the derived statistics (not required)

Extended Data Fig. 3 is reproduced from released statistics:

- ED 3: 2D histograms produced by
  `scripts/figures/figure_ed_joint/compute_joint_full.py` (full test set:
  17,519 hourly HK fields; 5,840 global snapshots).

Recomputing them requires the hindcast and the full sampling runs (~150 GB of
samples); the scripts are included for provenance but are not part of the
release acceptance test.

## 4. Expected values (sanity checks)

| Quantity | Paper value |
|---|---|
| HK Hs RMSE (test, 10-member mean) | 0.055 m (Fig. 3a annotation) |
| Global Hs RMSE (test) | 0.204 m |
| Fig. 2a HK case peak Hs | 5.29 m |
| Fig. 5g Kompasu valid time | 13 Oct 2021 00Z |
| Fig. 5h/i Ian valid time | 28 Sep 2022 12Z |
| Fig. 6 Ragasa peak | 24 Sep 2025 03Z |
| ED 5 storm / typical | 1 Jul 2022 20Z / 25 Jan 2021 17Z |
| ED 6 storm / typical | 24 Sep 2022 06Z / 15 Jan 2022 15Z |

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `no URL configured ...` | the Zenodo record is not public yet; set the URLs in `reproduce/downloads.yaml` or use your local archive |
| `BAD SHA256` | re-download; do not edit the archive |
| `missing ... samples` | restore the paper data tree at the repository root; `setup` does not download it |
| `missing ... hindcast` | see §1 and `DATA.md`; set `WF_GT_*` |
| map text differs from the paper | make sure `scripts/figures/fonts/` contains the four TeX Gyre Heros OTFs and matplotlib has a clean font cache (`rm -rf ~/.cache/matplotlib`) |
| `cartopy` attempts to download data offline | pre-fetch Natural Earth/GSHHS (§0) |
| GPU out of memory | reduce `sampling.batch_size` (`-o` is already exposed in the printed command) |
| CPU sampling refused | expected; exact reproduction requires a GPU |

## 6. Verification tolerances

`python reproduce.py verify` compares `reproduced/*.png` with `reference/*.png`.

- Pass: mean absolute pixel difference ≤ 1/255 **and** at most 0.1 % of pixels
  differ by more than 8/255.
- SSIM and per-panel metrics are written to
  `reproduced/verification_report.json`.
- Inference-tier outputs are stochastic; compare the printed statistics /
  ensemble means rather than pixels (see `WEIGHTS.md`).
