# Third-party data and attribution

The release bundles small derived products and open-data extracts. The original
sources and their terms are listed below. Data files can be removed if a
redistribution right is disputed; the affected panels then fall back to the
download/build scripts noted in `DATA.md`.

## Copernicus / ERA5

Contains modified Copernicus Climate Change Service information
(ERA5 reanalysis, 0.25°). Neither the European Commission nor ECMWF is
responsible for any use that may be made of the information it contains.
Files: `data/era5/npz_files/...`, `data/era5/global_npy_files_preinterpolated_daily/...`.

## ECMWF / WeatherBench2

IFS ENS forecast fields are derived from ECMWF open data and the
WeatherBench2 archive. Attribution: ECMWF, WeatherBench2 (Rasp et al. 2024).
Files: optional forcing archive only (the paper figures use the derived
samples, which are the authors’ model output). GenCast AI forecasts, when used
as forcing, are generated from the public GenCast/GraphCast weights and
WeatherBench2 inputs; follow the GraphCast model licence and WeatherBench2
terms.

## CMEMS

CMEMS L3 along-track SWH (product `WAVE_GLO_PHY_SWH_L3_NRT_014_001`) is used by
the satellite validation (not part of the panels in this release). Not bundled.

## Hong Kong Observatory (HKO) / IOC-VLIZ

Tide-gauge records (IOC Sea Level Station Monitoring Facility) and observed
daily tidal extrema / flooding reports are used in Fig. 6b,d,e under HKO and
IOC/VLIZ terms. Files: `data/hk_coastline/tide_gauge_ragasa/*`,
`data/hk_coastline/ragasa_hko_records.json`.

## Hong Kong Lands Department (LandsD)

DTM (5 m), orthophoto tiles and coastline geometry are used in Fig. 6a,b under
the Hong Kong Government open-data licence (attribution). Files:
`data/hk_coastline/orthophoto/*`, `data/hk_coastline/dtm_landsd_official/*`,
`data/hk_coastline/hk_coastline_segments.csv` (built from OSM + LandsD).
Rebuild scripts: `scripts/fetch_landsd_dtm.py`,
`scripts/fetch_landsd_orthophoto.py`, `scripts/fetch_landsd_coastline.py`,
`scripts/build_hk_coastline_segments.py`.

## OpenStreetMap (ODbL)

The coastline segment table derives from OpenStreetMap geometry.
© OpenStreetMap contributors, ODbL.

## GSHHS / NOAA

Global Self-consistent Hierarchical High-resolution Shorelines (Wessel &
Smith) are fetched by cartopy at runtime; public domain. Not bundled.

## Natural Earth

1:10m / 1:50m / 1:110m land and coastline layers are fetched by cartopy at
runtime; public domain. Not bundled.

## TeX Gyre Heros

The bundled fonts are under the GUST Font License (GUST, Polish TeX Users
Group); see `scripts/figures/fonts/`.
