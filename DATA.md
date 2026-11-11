# Data sources and access

This release redistributes **code, trained weights and model assets**. It does
**not** redistribute the hindcast targets or the figure source data; the latter
are provided as Source Data with the paper. All other inputs are public or open
government data.

## 1. Hindcast targets (not redistributed)

The Hong Kong and global wave/water-level hindcasts are third-party products.
They are used to train the released models and to draw the reference fields.

- Provider: TODO
- Access: TODO
- Contact: TODO

> The Hong Kong and global hindcast targets are third-party products and cannot be redistributed with this release.

For five panels (Fig. 5g–i, ED 5, ED 6) only a few reference frames are needed;
see `REPRODUCE.md` §1 for the exact files and the `WF_GT_*` overrides. If you
have access to the full hindcast, the original training/evaluation layout is:

- Hong Kong: `data/hindcast/processed_npz/wave_<YYYYMMDDTHH>.npz`
  (keys `hs`, `tp`, `dm`, `elev`, shape `(7788,)`), hourly 1970–2022.
- Global: `data/global_hindcast/npz_files/<yyyy>/global_hindcast_<yyyymmdd>_<hh>.npz`
  (key `hs`, shape `(651652,)`), 3-hourly 2021–2022.
- Global four-variable: `data/global_hindcast/npy_files_daily_4var/<yyyy>/global_hindcast_<yyyymmdd>.npy`
  (shape `(8, 651652, 4)`), daily packs.

The release's training split (1970–2018 / 2019–2020 validation / 2021–2022
test) follows the paper’s Methods.

## 2. Atmospheric forcing (public)

| Source | Use | Access | License |
|---|---|---|---|
| ERA5 (0.25°, hourly, 4 surface variables) | training forcing; Fig. 2a schematic; inference forcing for 2021/2022/2025 cases | Copernicus Climate Data Store, https://cds.climate.copernicus.eu | Copernicus licence (attribution) |
| IFS ENS / HRES (0.25°, 6-hourly, 50 members) | Fig. 5 forecast forcing | ECMWF open data https://data.ecmwf.int/forecasts/ and WeatherBench2 `gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr` | ECMWF licence (attribution) |
| GenCast (0.25° ensemble) | comparison forcing in Fig. 5 | WeatherBench2 / GraphCast public weights | CC BY 4.0 (model) / WeatherBench2 terms |

The release includes small ERA5 slices (a single 1970 frame, the Ragasa 2025
window, two 2021/2022 case windows, four global pre-interpolated days) so the
render and small inference recipes work offline. Full forcing can be
regenerated with `scripts/download_era5_*.py` / the scripts in
`scripts/live/` (not included) or the CDS API.

## 3. Observations and coastal assets (public / open government)

| Source | Use | Access | License / attribution |
|---|---|---|---|
| CMEMS L3 along-track SWH (Sentinel-6A etc.) | ED 1 / SI satellite validation (not in this release’s panels) | Copernicus Marine Service | Copernicus Marine licence |
| HKO tide gauges (Quarry Bay, Shek Pik, Tai Miu Wan, Tai Po Kau) and flooding reports | Fig. 6b,d,e | HKO / IOC Sea Level Station Monitoring Facility | HKO terms; IOC/VLIZ terms |
| LandsD DTM (5 m), orthophoto tiles, coastline geometry | Fig. 6a,b | Hong Kong Lands Department GeoData Store | HK Government open data licence (attribution) |
| OSM coastline (base for segment construction) | Fig. 6 assets | OpenStreetMap | ODbL |
| GSHHS full-resolution shorelines | Fig. 2a, ED 5/6 maps | NOAA / SOEST | public domain |
| Natural Earth (10m/50m/110m) | all maps | naturalearthdata.com | public domain |

The derived overtopping products (`data/hk_coastline/overtopping/*`) and the
segment table were produced by the authors from these open datasets; per-file
provenance is listed in `THIRD_PARTY.md` and in the Source Data manifest.

## 4. What each panel needs

The inputs below describe the **Source Data archive provided with the paper**.
They are not downloaded by this repository; unpack that archive at the
repository root to use the figure-rendering commands.

| Panel | Source Data inputs | User-supplied |
|---|---|---|
| Fig. 2a | mesh coordinates, ERA5 1970 frame, AI ocean mask, GSHHS | – |
| Fig. 3d,e,f | coordinates, clusters, 10 global samples (2022-09-24 00Z), ocean geometry | – |
| Fig. 5g | 4 lead × 10 HK samples (bc2) | Kompasu GT frame |
| Fig. 5h,i | 4 lead × 10 global samples | Ian GT frame |
| Fig. 6 | q products (analysis + forecast overlay), segments, HKO records, orthophoto/DTM caches, 100 peak samples, forecast samples | – |
| ED 2a | mesh coordinates, coastlines | – |
| ED 2b | four per-node squared-error arrays | – |
| ED 3 | six pairwise 2D histograms (HK/global) | – |
| ED 5 | 2×10 HK samples, HK Hs time-series CSV | 2 HK GT frames |
| ED 6 | 2×10 global four-variable samples | 2 global daily GT packs |

## 5. Checksums and manifests

The weight files are hosted on [Hugging Face](https://huggingface.co/tikitaka0345/WaveFlow); the
[Zenodo record](https://zenodo.org/records/24127153) carries the DOI and the per-file checksums.
The figure source data are provided as Source Data with the paper and are not
hosted here.

- `weights/MANIFEST.json`: per-weight SHA256, parameter count, source run.
- `reproduce/downloads.yaml`: archive URLs and SHA256.
- The Source Data archive contains its own manifest with per-file SHA256,
  size, source and license; it is not part of this repository.
