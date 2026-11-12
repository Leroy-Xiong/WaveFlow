# Normalization statistics

The checkpoints expect the same normalization that was used for training. The
statistics are shipped in `assets.tar.gz` by `python reproduce.py setup
--weights`; they are also listed here so users can inspect or reimplement the
transforms without reading the checkpoints.

## Formula

| Variable | Transform |
|---|---|
| `hs`, `tp` | `(log1p(x) - log_mean) / log_std` |
| `dm` (mean direction) | `sin(deg * pi / 180)`, `cos(deg * pi / 180)` |
| `elev` | `(x - mean) / std` |
| ERA5 `u10`, `v10`, `msl` | min-max to `[-1, 1]`: `2 * (x - min) / (max - min) - 1` |

## Global Hs model (`waveflow-global-hs-ep99.pt`)

Files:

- `data/global_hindcast/statistics.json`
- `data/era5/era5_statistics_global.json`

| Field | Value |
|---|---|
| `hs_log_mean` | `0.6469168089567439` |
| `hs_log_std` | `0.471009222765671` |
| `hs` min / max | `0.0` / `33.48505783081055` |
| ERA5 `u10` min / max | `-97.96321868896484` / `76.59862518310547` |
| ERA5 `v10` min / max | `-99.54826354980469` / `99.84466552734375` |
| ERA5 `msl` min / max | `90007.8515625` / `107105.296875` |

## Global four-variable model (`waveflow-global-4var-ep199.pt`)

File: `data/global_hindcast/statistics_4var.json`

| Field | Value |
|---|---|
| `hs_log_mean` | `0.6473872694388713` |
| `hs_log_std` | `0.4709545400605686` |
| `tp_log_mean` | `1.7516102144848213` |
| `tp_log_std` | `0.5462238820847745` |
| `elev_mean` | `5.8602038639777195e-05` |
| `elev_std` | `0.10928607851676901` |

ERA5 normalization for this model uses
`data/era5/era5_statistics_global.json` (same global min/max as above).

## Hong Kong four-variable model (`waveflow-hk-4var-ep49.pt`)

Files:

- `data/hindcast/wave_data_stats.json`
- `data/era5/era5_statistics.json`

| Field | Value |
|---|---|
| `hs_log_mean` | `0.36917033791542053` |
| `hs_log_std` | `0.27505195140838623` |
| `tp_log_mean` | `1.7348815202713013` |
| `tp_log_std` | `0.47826823592185974` |
| `elev_mean` | `0.10890812124655987` |
| `elev_std` | `0.6584666394000654` |
| ERA5 `u10` min / max | `-30.974777221679688` / `28.76935577392578` |
| ERA5 `v10` min / max | `-29.52313804626465` / `31.740633010864258` |
| ERA5 `msl` min / max | `96123.1796875` / `104037.0078125` |

## Notes

- `msl` is stored in Pa; do not convert to hPa before applying the min-max
  transform.
- `dm` is encoded as two channels, so the normalized tensor has one more
  channel than the raw wave tensor.
- If a statistics file is missing, run:

  ```bash
  python reproduce.py setup --weights
  ```

  The `assets.tar.gz` archive contains all files listed on this page.
