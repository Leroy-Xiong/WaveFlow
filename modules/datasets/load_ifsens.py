import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.utils import ERA5_HK_LON_RANGE_1, ERA5_HK_LAT_RANGE_1, ERA5_VARIABLE_NAMES


class IFSENSProcessor:
    def __init__(self):
        self.dataset = xr.open_zarr('gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr')

    def export_ens_forecast(self, output_dir='./data/ifs/ens', start_time='2021-01-01', end_time='2022-12-31'):
        """
        Export ENS forecast data for specified time range.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the exported files.
        start_time : str
            Start time for data extraction (e.g., '2021-01-01').
        end_time : str
            End time for data extraction (e.g., '2022-12-31').
        """
        print(f"Dataset structure:\n{self.dataset}")
        
        # Select variables
        data_subset = self.dataset[ERA5_VARIABLE_NAMES]
        
        # Select spatial region
        # ENS longitude is 0-360, latitude is -90 to 90
        lon_mask = (data_subset.longitude >= ERA5_HK_LON_RANGE_1[0]) & (data_subset.longitude < ERA5_HK_LON_RANGE_1[1])
        lat_mask = (data_subset.latitude >= ERA5_HK_LAT_RANGE_1[0]) & (data_subset.latitude < ERA5_HK_LAT_RANGE_1[1])
        
        subset = data_subset.sel(
            longitude=data_subset.longitude[lon_mask],
            latitude=data_subset.latitude[lat_mask]
        )
        
        # Select time range
        subset = subset.sel(time=slice(start_time, end_time))
        
        all_times = subset.time.values
        total_files = len(all_times)
        
        print(f"Starting export of {total_files} ENS forecast files from {start_time} to {end_time}")
        print(f"Output directory: {output_dir}")
        
        # Process in chunks to avoid memory issues
        chunk_size = 100 
        
        for i in tqdm(range(0, len(all_times), chunk_size), desc="Processing chunks"):
            # Get time chunk
            time_chunk = all_times[i:i+chunk_size]
            
            # Load chunk into memory
            ds_chunk = subset.sel(time=time_chunk).compute()
            
            # Process each time point individually
            for time_point in time_chunk:
                # Select data for single time point (initialization time)
                ds_single_time = ds_chunk.sel(time=time_point)
                
                # Get initialization time
                ts = pd.to_datetime(time_point)
                
                # Iterate over prediction timedeltas
                for td in ds_single_time.prediction_timedelta.values:
                    # Select data for specific prediction step
                    ds_step = ds_single_time.sel(prediction_timedelta=td)
                    
                    # Calculate hours for folder naming
                    hours = int(pd.Timedelta(td).total_seconds() / 3600)
                    lead_time_dir = f"lead_time_{hours}h"
                    
                    # Create directory structure: output_dir/lead_time_Xh/YYYY/
                    year_dir = os.path.join(output_dir, lead_time_dir, ts.strftime('%Y'))
                    os.makedirs(year_dir, exist_ok=True)
                    
                    # Construct filename
                    time_str = ts.strftime('%Y%m%dT%H')
                    filename = f"ifs_ens_{time_str}.nc"
                    filepath = os.path.join(year_dir, filename)
                    
                    # Set compression encoding
                    encoding = {
                        var: {'zlib': True, 'complevel': 4}
                        for var in ERA5_VARIABLE_NAMES
                    }
                    
                    # Save to netCDF file
                    ds_step.to_netcdf(filepath, encoding=encoding, engine='h5netcdf')
        
        print(f"\nExport completed! Files saved to {output_dir}")

    def export_ens_forecast_global(
        self,
        output_dir='./data/ifs/ens_global',
        start_time='2020-12-15',
        end_time='2022-12-31',
        lead_times_hours=(12, 24, 48, 72),
        variables=None,
        max_workers=8,
        skip_existing=True,
    ):
        """
        Download IFS-ENS forecast data on the full global grid for the requested
        lead times, writing one NetCDF per (init_time, lead_time).

        The directory layout mirrors `export_ens_forecast`:
            output_dir/
                lead_time_<H>h/
                    <YYYY>/
                        ifs_ens_<YYYYMMDDTHH>.nc

        Parameters
        ----------
        output_dir : str
            Root output directory. A separate root is recommended so that the
            HK-cropped dataset under ``./data/ifs/ens`` is not overwritten.
        start_time, end_time : str
            Init-time range, inclusive (e.g. '2020-12-15', '2022-12-31'). The
            zarr cube already stores init times every 12h (00/12 UTC), so no
            extra subsampling is needed.
        lead_times_hours : sequence of int
            Forecast lead times (in hours) to keep. Default keeps the lead
            times used by the HK model evaluation (12/24/48/72h).
        variables : list of str, optional
            Variables to export. Defaults to ``ERA5_VARIABLE_NAMES``
            (u10/v10/mslp) to stay aligned with the HK pipeline.
        max_workers : int
            Number of parallel threads for downloading. Reading from GCS is
            I/O-bound, so threads are well suited.
        skip_existing : bool
            If True, skip an init-time when all of its lead-time files already
            exist on disk.

        Returns
        -------
        int
            Number of init times processed (i.e. with at least one file
            written or successfully skipped).
        """
        if variables is None:
            variables = list(ERA5_VARIABLE_NAMES)

        lead_times_hours = list(lead_times_hours)
        # Map hours -> numpy.timedelta64('ns'), matching the zarr coord dtype.
        leads_td = [np.timedelta64(int(h), 'h').astype('timedelta64[ns]')
                    for h in lead_times_hours]

        # Restrict data variables and lead times up front so the lazy view is small.
        data_subset = (
            self.dataset[variables]
            .sel(prediction_timedelta=leads_td)
            .sel(time=slice(start_time, end_time))
        )

        all_times = data_subset.time.values
        if len(all_times) == 0:
            raise ValueError(
                f"No init times in the IFS-ENS dataset within {start_time}..{end_time}"
            )

        # Pre-create lead-time folders (year subfolders are created lazily).
        for h in lead_times_hours:
            os.makedirs(os.path.join(output_dir, f'lead_time_{h}h'), exist_ok=True)

        encoding = {var: {'zlib': True, 'complevel': 4} for var in variables}

        def _file_path_for(ts: pd.Timestamp, h: int) -> str:
            return os.path.join(
                output_dir, f'lead_time_{h}h', ts.strftime('%Y'),
                f'ifs_ens_{ts.strftime("%Y%m%dT%H")}.nc',
            )

        def _process_one_init(time_point) -> tuple:
            """Download & write all requested lead times for a single init time."""
            ts = pd.to_datetime(time_point)
            target_paths = [_file_path_for(ts, h) for h in lead_times_hours]

            # Resume support: skip when all leads already exist.
            if skip_existing and all(os.path.exists(p) for p in target_paths):
                return ts, 0, 0  # written=0, skipped=len(leads)

            ds_init = data_subset.sel(time=time_point).compute()

            written = 0
            for h, path in zip(lead_times_hours, target_paths):
                if skip_existing and os.path.exists(path):
                    continue
                td = np.timedelta64(int(h), 'h').astype('timedelta64[ns]')
                ds_step = ds_init.sel(prediction_timedelta=td)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # Atomic-ish write: write to .tmp then rename.
                tmp_path = path + '.tmp'
                ds_step.to_netcdf(tmp_path, encoding=encoding, engine='h5netcdf')
                os.replace(tmp_path, path)
                written += 1
            return ts, written, len(lead_times_hours) - written

        print(
            f"Exporting global IFS-ENS forecasts: "
            f"{len(all_times)} init times x {len(lead_times_hours)} leads "
            f"({lead_times_hours} h), variables={variables}"
        )
        print(f"Output directory: {output_dir}")
        print(f"Parallel threads: {max_workers}, skip_existing={skip_existing}")

        n_done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one_init, t): t for t in all_times}
            with tqdm(total=len(futures), desc='Global IFS-ENS download') as pbar:
                for fut in as_completed(futures):
                    try:
                        ts, written, skipped = fut.result()
                        n_done += 1
                        pbar.set_postfix(last=str(ts)[:13],
                                         w=written, s=skipped)
                    except Exception as e:
                        ts = futures[fut]
                        tqdm.write(f"[error] init={ts}: {e}")
                    finally:
                        pbar.update(1)

        print(f"\nGlobal export completed: {n_done}/{len(all_times)} init times. "
              f"Files under {output_dir}")
        return n_done

