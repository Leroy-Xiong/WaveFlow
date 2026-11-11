## Data availability (draft for the manuscript)

The ERA5 reanalysis used for atmospheric forcing is available from the
Copernicus Climate Change Service (https://cds.climate.copernicus.eu). The IFS
ENS and HRES forecast fields are available from ECMWF open data
(https://data.ecmwf.int/forecasts/) and the WeatherBench2 archive; GenCast AI
forecast fields can be generated from the public GraphCast/GenCast weights and
WeatherBench2 initial conditions. The CMEMS L3
along-track significant wave height is available from the Copernicus Marine
Service (product WAVE_GLO_PHY_SWH_L3_NRT_014_001). Tide-gauge records and
flooding reports are available from the Hong Kong Observatory and the IOC Sea
Level Station Monitoring Facility; the DTM, orthophoto and coastline assets are
available from the Hong Kong Lands Department GeoData Store. The Hong Kong and
global wave hindcast targets are a third-party product and are not
redistributable; requests should be directed to TODO.

## Code availability (draft for the manuscript)

WaveFlow code, training and sampling configs, the trained weights and the
derived figure source data are available at https://github.com/Leroy-Xiong/WaveFlow (code: MIT licence;
weights: CC BY 4.0), with the model weights mirrored at https://huggingface.co/tikitaka0345/WaveFlow and
the dataset record at 10.5281/zenodo.24106732. The figure source data are provided as
Source Data with the paper and are not part of the public repository. The
release reproduces the panels listed in its README; hindcast-dependent
reference fields require the third-party hindcast described above.
