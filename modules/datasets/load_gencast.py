"""
GenCast forecast data processor.

Extracts HRES-fc0 analysis from WeatherBench2 GCS zarr, runs GenCast Operational
model inference, interpolates to 6h intervals (max 72h), and exports u10/v10/mslp
in project's standard NetCDF format.

Pipeline:
  1. Pre-download HRES analysis fields for 2021-2022 init times (00Z/12Z)
  2. Build GenCast-compatible input (merge HRES atmos + ERA5 SST + static vars)
  3. Run GenCast Operational model inference (12h steps)
  4. Interpolate to 6h intervals, max 72h (init analysis prepended as t=0)
  5. Export PER-MEMBER files lead_time_XXh/YYYY/gencast_m{MM}_YYYYMMDDTHH.nc
     (member-forced WaveFlow sampling; lead 0 = init analysis, hard-linked)

Usage (figure5: <2019 weights, 10 members, 72h):
  proc = GenCastProcessor(model_variant='0p25deg_2019')
  proc.run_batch_inference(start='20210101', end='20221231',
                           num_ensemble_members=10, forecast_steps=6,
                           crop_output_dir='./data/gencast_members_hk',
                           delete_input_cache=True)
"""

import logging
import os
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Sequence, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data sources
HRES_ANALYSIS_ZARR = 'gs://weatherbench2/datasets/hres_t0/forecasts_t0_analysis_2016-2022-6h-0p25deg-chunk-1.zarr'
ERA5_STATIC_ZARR = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'

# Model variants
GENCast_MODEL_VARIANTS = {
    '0p25deg_2019': 'GenCast 0p25deg <2019.npz',
    '0p25deg_operational': 'GenCast 0p25deg Operational <2022.npz',
}

GENCast_GCS_BUCKET = 'dm_graphcast'
GENCast_GCS_PREFIX = 'gencast'

# GenCast pressure levels (subset of 37 ERA5 levels)
GENCast_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# HRES -> GenCast variable mapping
HRES_SURFACE_VARS = [
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '2m_temperature', 'mean_sea_level_pressure', 'total_precipitation_6hr',
]

HRES_PRESSURE_VARS = [
    'geopotential', 'specific_humidity', 'temperature',
    'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
]

# Output subset (matches project ERA5_VARIABLE_NAMES)
OUTPUT_VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']


def _xarray_vmap(fn, dim: str):
    """xarray_jax.pmap 的 vmap 版（in_axes/out_axes=0，首维映射 dim 轴）。

    graphcast 的 rollout 只提供 pmap 路径（按 device_count 串行切样本），单卡
    上 10 成员会被拆成 10 次串行 rollout；此包装与 xarray_jax.pmap 结构一致，
    但用 jax.vmap 把整块样本一次前向（数值等价，见 run_single_inference）。
    """
    import jax
    from graphcast import xarray_jax

    input_treedef = None
    output_treedef = None

    def fn_passed_to_vmap(*flat_args):
        assert input_treedef is not None

        def check_and_remove_leading_dim(dims):
            try:
                index = dims.index(dim)
            except ValueError:
                index = None
            if index != 0:
                raise ValueError(f'Expected dim {dim} at index 0, found at {index}.')
            return dims[1:]
        with xarray_jax.dims_change_on_unflatten(check_and_remove_leading_dim):
            args = jax.tree_util.tree_unflatten(input_treedef, flat_args)
        result = fn(*args)
        nonlocal output_treedef
        flat_result, output_treedef = jax.tree_util.tree_flatten(result)
        return flat_result

    vmapped_fn = jax.vmap(fn_passed_to_vmap, in_axes=0, out_axes=0)

    def result_fn(*args):
        nonlocal input_treedef
        flat_args, input_treedef = jax.tree_util.tree_flatten(args)
        flat_result = vmapped_fn(*flat_args)
        assert output_treedef is not None
        with xarray_jax.dims_change_on_unflatten(lambda dims: (dim,) + dims):
            return jax.tree_util.tree_unflatten(output_treedef, flat_result)

    return result_fn


def _replicate_sample_ds(ds, n: int):
    """沿新 'sample' 首维复制 n 份（rollout._replicate_dataset 的 numpy 版，
    不依赖 devices 列表；成员无关的 inputs/forcings/template 用）。"""
    from graphcast import xarray_jax

    def rep_var(var):
        if 'sample' in var.dims:
            return var.transpose('sample', ...)
        data = np.stack(n * [var.data], axis=0)
        return xarray_jax.Variable(data=data, dims=('sample',) + var.dims,
                                   attrs=var.attrs)

    if ds is None:
        return None
    data_vars = {name: rep_var(var)
                 for name, var in ds.data_vars.variables.items()}
    coords = {name: c.variable for name, c in ds.coords.items()}
    return xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)


class GenCastProcessor:
    """GenCast forecast processor using HRES analysis + Operational model."""

    def __init__(
        self,
        model_variant: str = '0p25deg_operational',
        weights_dir: str = './baseline_repros/competitor_analysis/gencast_weights',
        output_dir: str = './data/gencast_members',
        cache_dir: str = './data/gencast_cache',
        bfloat16: bool = False,
        halve_tiles: bool = False,
    ):
        if model_variant not in GENCast_MODEL_VARIANTS:
            raise ValueError(f"Unknown variant. Options: {list(GENCast_MODEL_VARIANTS)}")
        self.model_variant = model_variant
        self.weights_file = GENCast_MODEL_VARIANTS[model_variant]
        self.weights_dir = Path(weights_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # GPU 兼容选项(默认关闭,48GB 卡上试验过均无效/更差,见 _load_model):
        # bfloat16: graphcast 官方 Bfloat16Cast 混合精度;halve_tiles: 注意力
        # tile 块减半(数学等价)。0.25° 模型的 f32 固定缓冲(quadsplit 3.11M
        # 网格节点 × 1536 通道 × 4B ≈ 17.8GB)+ ~35GB 基线,48GB 卡无解,
        # 需要 ≥80GB 卡。
        self.bfloat16 = bfloat16
        self.halve_tiles = halve_tiles

        # Lazy-loaded
        self._params = None
        self._task_config = None
        self._sampler_config = None
        self._noise_config = None
        self._noise_encoder_config = None
        self._denoiser_architecture_config = None
        self._diffs_stddev_by_level = None
        self._mean_by_level = None
        self._stddev_by_level = None
        self._min_by_level = None

    # ------------------------------------------------------------------
    # Step 0: Download model weights
    # ------------------------------------------------------------------

    def download_weights(self, force: bool = False) -> Path:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token='anon')

        wp = self.weights_dir / self.weights_file
        if not wp.exists() or force:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            gcs = f'{GENCast_GCS_BUCKET}/{GENCast_GCS_PREFIX}/params/{self.weights_file}'
            logger.info(f"Downloading: {gcs}")
            fs.get(gcs, str(wp))

        stats_dir = self.weights_dir / 'stats'
        stats_dir.mkdir(parents=True, exist_ok=True)
        for sf in ['diffs_stddev_by_level.nc', 'mean_by_level.nc',
                    'stddev_by_level.nc', 'min_by_level.nc']:
            sp = stats_dir / sf
            if sp.exists() and not force:
                continue
            fs.get(f'{GENCast_GCS_BUCKET}/{GENCast_GCS_PREFIX}/stats/{sf}', str(sp))
        return wp

    # ------------------------------------------------------------------
    # Step 1: Pre-download HRES analysis + build GenCast input cache
    # ------------------------------------------------------------------

    def pre_download_analysis(
        self,
        start_date: str = '20210101',
        end_date: str = '20221231',
        max_workers: int = 4,
        skip_existing: bool = True,
        forecast_steps: int = 6,
    ):
        """Pre-download all HRES analysis fields and build GenCast input cache.

        For each init time (00Z, 12Z), extracts HRES-fc0 analysis at T-12h and T,
        merges with ERA5 SST + static vars + target-lead template frames, and
        caches as NetCDF (see _build_and_cache_input for the frame layout).
        """
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')

        tasks = []
        d = start
        while d <= end:
            for h in [0, 12]:
                tasks.append(datetime(d.year, d.month, d.day, h))
            d += timedelta(days=1)

        logger.info(f"Pre-downloading {len(tasks)} init times ({start_date} - {end_date})")

        # Pre-load ERA5 static vars (same for all times)
        static_vars = self._load_static_vars()

        # Pre-load ERA5 SST (slowly varying, nearest-neighbor OK)
        era5_zarr = xr.open_zarr(ERA5_STATIC_ZARR, storage_options={"token": "anon"}, consolidated=False)
        # Open HRES analysis zarr once (consolidated metadata is huge over anon
        # GCS; consolidated=False reads only the per-variable attrs we need)
        hres_zarr = xr.open_zarr(HRES_ANALYSIS_ZARR, storage_options={"token": "anon"}, consolidated=False)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for dt in tasks:
                cache_path = self._cache_path(dt)
                if skip_existing and cache_path.exists():
                    continue
                fut = pool.submit(self._build_and_cache_input, dt, static_vars,
                                  era5_zarr, hres_zarr, forecast_steps)
                futures[fut] = dt

            with tqdm(total=len(futures), desc='Pre-download') as pbar:
                for fut in as_completed(futures):
                    dt = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        logger.error(f"Failed {dt}: {e}")
                    pbar.update(1)

        logger.info(f"Pre-download complete. Cached in {self.cache_dir}")

    def _cache_path(self, dt: datetime) -> Path:
        return self.cache_dir / f"gencast_input_{dt.strftime('%Y%m%dT%H')}.nc"

    def _load_static_vars(self) -> dict:
        """Load static fields (geopotential_at_surface, land_sea_mask) from ERA5.

        Cached locally (cache_dir/gencast_static.nc) after the first fetch —
        opening the full 37-level hourly ERA5 zarr just for two 2D fields costs
        minutes of metadata reads over throttled networks, once per shard."""
        local = self.cache_dir / 'gencast_static.nc'
        if local.exists():
            cached = xr.open_dataset(local)
            return {k: cached[k] for k in
                    ('geopotential_at_surface', 'land_sea_mask')}
        era5 = xr.open_zarr(ERA5_STATIC_ZARR, storage_options={"token": "anon"}, consolidated=False)
        # Static vars: no time dim (or take first timestep)
        z_surf = era5['geopotential_at_surface']
        lsm = era5['land_sea_mask']
        if 'time' in z_surf.dims:
            z_surf = z_surf.isel(time=0)
        if 'time' in lsm.dims:
            lsm = lsm.isel(time=0)
        static = {
            'geopotential_at_surface': z_surf.compute(),
            'land_sea_mask': lsm.compute(),
        }
        # Rename coords to match GenCast expectations
        for k in static:
            static[k] = static[k].rename({'latitude': 'lat', 'longitude': 'lon'})
            # Ensure lat ascending
            if float(static[k].lat[0]) > float(static[k].lat[-1]):
                static[k] = static[k].sortby('lat')
        xr.Dataset(static).to_netcdf(str(local) + '.tmp', engine='netcdf4')
        os.replace(str(local) + '.tmp', str(local))
        return static

    def _build_and_cache_input(
        self,
        init_dt: datetime,
        static_vars: dict,
        era5_zarr: xr.Dataset,
        hres_zarr: xr.Dataset = None,
        forecast_steps: int = 6,
    ):
        """Build GenCast input for a single init time and cache it."""
        cache_path = self._cache_path(init_dt)
        if cache_path.exists():
            return

        hres = hres_zarr if hres_zarr is not None else xr.open_zarr(
            HRES_ANALYSIS_ZARR, storage_options={"token": "anon"}, consolidated=False)

        # Cache layout required by extract_inputs_targets_forcings for the
        # autoregressive rollout (task_config.input_duration='24h' = 2 input
        # frames at 12h spacing; the time shift aligns the LAST frame with the
        # target_lead_times stop, so the dataset must span inputs + all target
        # leads, exactly like the official demo example files):
        #   [T-12h, T]                    real HRES-fc0 analysis (model inputs)
        #   [T+12h .. T+steps*12h]        template frames — target values are
        #                                 irrelevant (rollout uses a NaN template)
        #                                 but forcings (year/day progress) must be
        #                                 real; copy the T frame's fields.
        input_times = [init_dt - timedelta(hours=12), init_dt]
        template_times = [init_dt + timedelta(hours=12 * k)
                          for k in range(1, forecast_steps + 1)]
        # Need T-18h..T (extra 6h back so tp_12hr(T-12h) = tp6(T-12h)+tp6(T-18h))
        t_start = np.datetime64((init_dt - timedelta(hours=18)).strftime('%Y-%m-%dT%H:%M:%S'))
        t_end = np.datetime64(init_dt.strftime('%Y-%m-%dT%H:%M:%S'))

        # --- Surface vars from HRES (extra timesteps for precip computation) ---
        ds_surf = hres[HRES_SURFACE_VARS].sel(time=slice(t_start, t_end))
        ds_surf = ds_surf.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Convert total_precipitation_6hr -> total_precipitation_12hr
        # tp_6hr at time T is the 6h accumulation ending at T
        # tp_12hr(T) = tp_6hr(T) + tp_6hr(T-6h)
        tp6 = ds_surf['total_precipitation_6hr']
        tp12 = tp6 + tp6.shift(time=-1, fill_value=0)
        ds_surf['total_precipitation_12hr'] = tp12
        ds_surf = ds_surf.drop_vars('total_precipitation_6hr')

        # --- SST from ERA5 (nearest time, broadcast) ---
        era5_slice = era5_zarr['sea_surface_temperature'].sel(time=slice(t_start, t_end))
        era5_slice = era5_slice.rename({'latitude': 'lat', 'longitude': 'lon'})
        if float(era5_slice.lat[0]) > float(era5_slice.lat[-1]):
            era5_slice = era5_slice.sortby('lat')
        # Reindex to match HRES times
        sst = era5_slice.reindex(time=ds_surf.time, method='nearest')
        ds_surf['sea_surface_temperature'] = sst

        # --- Pressure level vars from HRES ---
        ds_plev = hres[HRES_PRESSURE_VARS].sel(time=slice(t_start, t_end))
        ds_plev = ds_plev.rename({'latitude': 'lat', 'longitude': 'lon'})
        # Select GenCast's 13 levels
        avail_levels = [int(l) for l in hres.level.values]
        gencast_levels = [l for l in GENCast_PRESSURE_LEVELS if l in avail_levels]
        ds_plev = ds_plev.sel(level=gencast_levels)

        # --- Ensure lat ascending ---
        if float(ds_surf.lat[0]) > float(ds_surf.lat[-1]):
            ds_surf = ds_surf.sortby('lat')
        if float(ds_plev.lat[0]) > float(ds_plev.lat[-1]):
            ds_plev = ds_plev.sortby('lat')

        # --- Resample to exact input times ---
        input_np = np.array([np.datetime64(t.strftime('%Y-%m-%dT%H:%M:%S')) for t in input_times])
        ds_surf = ds_surf.sel(time=input_np, method='nearest')
        ds_plev = ds_plev.sel(time=input_np, method='nearest')

        # --- Merge ---
        ds = xr.merge([ds_surf, ds_plev])

        # --- Append template frames at the target leads (T+12h..T+steps*12h):
        # copies of the T frame; values are unused (NaN template) but keep the
        # structure and (via the datetime coord) supply true progress forcings.
        last = ds.isel(time=-1)
        frames = [ds]
        for t in template_times:
            ft = np.datetime64(t.strftime('%Y-%m-%dT%H:%M:%S'))
            frames.append(last.expand_dims(time=[ft]))
        ds = xr.concat(frames, dim='time')

        # Add static vars
        for k, v in static_vars.items():
            ds[k] = v

        # time = offsets relative to the FINAL INPUT timestep (T): [-12h, 0,
        # +12h, ..., +steps*12h]. extract shifts/slices timedelta coords (demo
        # files use the same convention). Absolute times live in the separate
        # `datetime` coord, which add_derived_vars uses for progress features.
        all_times = input_times + template_times
        target_np = np.array([np.datetime64(t.strftime('%Y-%m-%dT%H:%M:%S')) for t in all_times])
        rel_time = np.array([np.timedelta64(int((t - input_times[-1]).total_seconds()), 's')
                             for t in all_times])
        ds = ds.assign_coords(time=('time', rel_time))
        ds = ds.expand_dims('batch')
        ds = ds.assign_coords(datetime=(('batch', 'time'), np.array([target_np])))

        # --- Cache (atomic + serialized: netCDF4/HDF5 writes are not thread-safe,
        # pre_download_analysis runs this in a thread pool) ---
        tmp = str(cache_path) + '.tmp'
        with GenCastProcessor._NETCDF_WRITE_LOCK:
            try:
                ds.to_netcdf(tmp, engine='netcdf4')
            except Exception:
                ds.to_netcdf(tmp, engine='h5netcdf')
        os.replace(tmp, str(cache_path))
        ds.close()

    # ------------------------------------------------------------------
    # Step 2: Load model
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._params is not None:
            return
        from graphcast import checkpoint, gencast

        wp = self.weights_dir / self.weights_file
        if not wp.exists():
            raise FileNotFoundError(f"Weights not found: {wp}. Run download_weights() first.")

        logger.info(f"Loading: {self.weights_file}")
        with open(wp, 'rb') as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)

        self._params = ckpt.params
        self._task_config = ckpt.task_config
        self._sampler_config = ckpt.sampler_config
        self._noise_config = ckpt.noise_config
        self._noise_encoder_config = ckpt.noise_encoder_config
        # Checkpoints ship with attention_type='splash_mha' (TPU-only kernels;
        # on GPU jax fails with "Scalar prefetch not supported in Triton
        # lowering"). triblockdiag_mha computes the same attention — this is
        # DeepMind's documented GPU path. The doc's ~60GB VRAM exceeds our 48GB
        # cards, so we additionally halve the attention tile blocks: they are
        # pure kernel-tiling parameters (mathematically identical output, more
        # kernel launches), and attention workspace scales with them.
        import dataclasses
        dac = ckpt.denoiser_architecture_config
        stc = dac.sparse_transformer_config
        if getattr(stc, 'attention_type', '') == 'splash_mha':
            kw = {'attention_type': 'triblockdiag_mha'}
            if self.halve_tiles:
                for f_ in ('block_q', 'block_kv', 'block_kv_compute',
                           'block_q_dkv', 'block_kv_dkv', 'block_kv_dkv_compute'):
                    v = getattr(stc, f_, None)
                    if v:
                        kw[f_] = max(64, int(v) // 2)
            dac = dataclasses.replace(
                dac, sparse_transformer_config=dataclasses.replace(stc, **kw))
            logger.info(f"attention splash_mha -> triblockdiag_mha (GPU path, "
                        f"halve_tiles={self.halve_tiles})")
        self._denoiser_architecture_config = dac
        logger.info(f"Model: {ckpt.description}")

        sd = self.weights_dir / 'stats'
        self._diffs_stddev_by_level = xr.load_dataset(sd / 'diffs_stddev_by_level.nc').compute()
        self._mean_by_level = xr.load_dataset(sd / 'mean_by_level.nc').compute()
        self._stddev_by_level = xr.load_dataset(sd / 'stddev_by_level.nc').compute()
        self._min_by_level = xr.load_dataset(sd / 'min_by_level.nc').compute()

    def _construct_wrapped_gencast(self):
        from graphcast import casting, gencast, normalization, nan_cleaning
        p = gencast.GenCast(
            sampler_config=self._sampler_config,
            task_config=self._task_config,
            denoiser_architecture_config=self._denoiser_architecture_config,
            noise_config=self._noise_config,
            noise_encoder_config=self._noise_encoder_config,
        )
        # bfloat16 混合精度(graphcast 官方 Bfloat16Cast 包装,默认关):
        # 实测 48GB 卡上 XLA 融合反而请求更大的缓冲(29.7GB vs f32 17.8GB),
        # 仅作保留选项。
        if self.bfloat16:
            p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p, diffs_stddev_by_level=self._diffs_stddev_by_level,
            mean_by_level=self._mean_by_level, stddev_by_level=self._stddev_by_level)
        p = nan_cleaning.NaNCleaner(
            p, reintroduce_nans=True, fill_value=self._min_by_level,
            var_to_clean='sea_surface_temperature')
        return p

    # ------------------------------------------------------------------
    # Step 3: Run inference for a single init time
    # ------------------------------------------------------------------

    def run_single_inference(
        self,
        init_dt: datetime,
        num_ensemble_members: int = 50,
        forecast_steps: int = 30,
        cache_predictions: bool = True,
        member_chunk: int = 2,
        batch_members: bool = False,
        steps_per_chunk: int = 1,
    ) -> xr.Dataset:
        """Run GenCast inference for one init time. Uses pre-cached input.

        Members are rolled out in chunks of `member_chunk`: activations scale
        with the per-call sample count, and one 10-member call needs ~18GB of
        extra VRAM (OOM on 48GB cards even with halved attention tiles).

        batch_members=True（2026-10-15 加，vmap 批量路径）: 单卡时用 jax.vmap
        把整块成员一次前向，替代 rollout 外层按 device_count 的串行循环——
        数值上与逐成员等价（相同 fold_in key 序列 + jax.random.split 对批量轴
        逐元素），预计 3-6× 提速。显存按成员数线性增长，OOM 时调小 member_chunk。

        Args:
            init_dt: Init time (00Z/12Z).
            num_ensemble_members: Ensemble size.
            forecast_steps: Number of 12h autoregressive steps.
            cache_predictions: Write the full-member prediction stack to
                cache_dir/gencast_pred_*.nc. The full-variable stack is ~2.4 GB per
                member-step set (all vars/levels); batch runs pass False and feed
                the in-memory predictions straight into export_to_project_format.
            member_chunk: Members per rollout call (1..num_ensemble_members).
            batch_members: 单卡 vmap 批量（绕过 pmap 串行）。
        """
        import jax
        import haiku as hk
        import dataclasses
        from graphcast import rollout, xarray_jax, data_utils

        self._load_model()
        cache_path = self._cache_path(init_dt)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}. Run pre_download_analysis first.")

        ds_input = xr.load_dataset(cache_path).compute()
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds_input,
            target_lead_times=slice('12h', f'{forecast_steps * 12}h'),
            **dataclasses.asdict(self._task_config),
        )

        @hk.transform_with_state
        def run_forward(i, tt, f):
            return self._construct_wrapped_gencast()(i, targets_template=tt, forcings=f)

        run_forward_jitted = jax.jit(
            lambda rng, i, t, f: run_forward.apply(self._params, {}, rng, i, t, f)[0])

        devices = jax.local_devices()
        n_dev = len(devices)
        use_vmap = batch_members and n_dev == 1

        all_rngs = [jax.random.fold_in(jax.random.PRNGKey(0), i)
                    for i in range(num_ensemble_members)]
        member_chunk = max(1, min(member_chunk, num_ensemble_members))
        parts = []
        for c0 in range(0, num_ensemble_members, member_chunk):
            m = min(member_chunk, num_ensemble_members - c0)
            if use_vmap:
                # vmap 批量：整块成员一次前向（单组直接调内层 generator，
                # 跳过 multiple_runs 的按设备串行循环）
                vmap_fn = _xarray_vmap(run_forward_jitted, dim='sample')

                def predictor(rng, inputs, targets_template, forcings):
                    return vmap_fn(rng, inputs, targets_template, forcings)

                rngs = np.stack(all_rngs[c0:c0 + m], axis=0)      # (m, 2)
                inputs_s = _replicate_sample_ds(inputs, m)
                tt_s = _replicate_sample_ds(targets * np.nan, m)
                forcings_s = _replicate_sample_ds(forcings, m)
                chunks = []
                gen = rollout.chunked_prediction_generator(
                    predictor_fn=predictor, rng=rngs, inputs=inputs_s,
                    targets_template=tt_s, forcings=forcings_s,
                    num_steps_per_chunk=steps_per_chunk)
                for chunk in tqdm(gen,
                                  total=max(1, forecast_steps // steps_per_chunk),
                                  desc=f'GenCast {init_dt:%Y%m%dT%H} vmap m{c0:02d}-{c0+m-1:02d}',
                                  leave=False):
                    chunk = chunk.assign_coords(
                        sample=np.arange(c0, c0 + m))
                    chunks.append(chunk)
                part = xr.combine_by_coords(chunks)
            else:
                run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim='sample')
                eff_ens = ((member_chunk + n_dev - 1) // n_dev) * n_dev
                rngs = np.stack(all_rngs[c0:c0 + member_chunk] +
                                [jax.random.fold_in(jax.random.PRNGKey(1), c0 + j)
                                 for j in range(member_chunk, eff_ens)], axis=0)
                tt = targets * np.nan
                chunks = []
                gen = rollout.chunked_prediction_generator_multiple_runs(
                    predictor_fn=run_forward_pmap, rngs=rngs, inputs=inputs,
                    targets_template=tt, forcings=forcings,
                    num_steps_per_chunk=steps_per_chunk,
                    num_samples=eff_ens, pmap_devices=devices)
                n_groups = (eff_ens + n_dev - 1) // n_dev
                for chunk in tqdm(gen,
                                  total=max(1, n_groups * forecast_steps // steps_per_chunk),
                                  desc=f'GenCast {init_dt:%Y%m%dT%H} m{c0:02d}-{c0+len(rngs)-1:02d}',
                                  leave=False):
                    chunks.append(chunk)
                part = xr.combine_by_coords(chunks).isel(
                    sample=slice(0, min(member_chunk, num_ensemble_members - c0)))
            parts.append(part)

        preds = xr.concat(parts, dim='sample') if len(parts) > 1 else parts[0]

        if cache_predictions:
            pred_path = self.cache_dir / f"gencast_pred_{init_dt.strftime('%Y%m%dT%H')}.nc"
            preds.to_netcdf(pred_path)
        return preds

    # ------------------------------------------------------------------
    # Step 4: Interpolate 12h -> 6h and export (per member)
    # ------------------------------------------------------------------

    def _t0_analysis_dataset(self, init_time_str: str) -> xr.Dataset:
        """Init-time HRES-fc0 analysis (the 3 OUTPUT_VARS, no time/batch/sample
        dims) from the last frame of the GenCast input cache - the t=0 state all
        ensemble members share."""
        input_path = self.cache_dir / f"gencast_input_{init_time_str}.nc"
        if not input_path.exists():
            raise FileNotFoundError(f"GenCast input cache not found: {input_path}")
        inputs = xr.load_dataset(input_path)
        t0 = inputs.isel(time=-1, batch=0)
        data_vars = {}
        for v in OUTPUT_VARS:
            arr = t0[v]
            for d in ['level', 'batch', 'sample']:
                if d in arr.dims and arr.sizes[d] == 1:
                    arr = arr.isel({d: 0})
            data_vars[v] = arr.astype(np.float32)
        return xr.Dataset(data_vars)

    def _interpolate_to_6h(self, predictions: xr.Dataset, max_lead_h: int = 72) -> xr.Dataset:
        """Interpolate GenCast predictions from 12h to 6h steps.

        xarray.interp is elementwise over all non-time dims, so this interpolates
        every member independently (no ensemble mean is taken here).

        `predictions` must include time=0 (the init analysis): GenCast targets
        start at 12h, so without the t0 point the 6h target would be a linear
        extrapolation below the source axis and come back NaN. See
        export_to_project_format, which prepends t0 before calling this.
        """
        target_hours = list(range(6, max_lead_h + 1, 6))
        target_times_ns = np.array(
            [np.timedelta64(int(h) * 3600 * 10**9, 'ns') for h in target_hours])
        return predictions.interp(time=target_times_ns, method='linear')

    # HDF5 (netcdf4 engine) is NOT thread-safe: concurrent writers corrupt global
    # state (segfaults / H5DSis_scale / file-lock errors). All NetCDF writes go
    # through this lock — the slow part of pipelines (network fetch) stays parallel.
    _NETCDF_WRITE_LOCK = threading.Lock()

    @classmethod
    def _atomic_save(cls, ds: xr.Dataset, fp: Path):
        """Atomic NetCDF write (.tmp + os.replace) so an interrupted job never
        leaves a truncated file that skip_existing would accept.

        Strips leftover non-index coords (scalar time/batch from the input cache,
        the unused 'level' coord) — h5netcdf chokes on orphan dimension scales."""
        fp.parent.mkdir(parents=True, exist_ok=True)
        ds = ds.reset_coords(drop=True)
        used_dims = {d for v in ds.data_vars.values() for d in v.dims}
        drop = [c for c, da in ds.coords.items() if set(da.dims) - used_dims]
        if drop:
            ds = ds.drop_vars(drop)
        tmp = str(fp) + '.tmp'
        encoding = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
        with cls._NETCDF_WRITE_LOCK:
            # netcdf4 preferred; h5netcdf fallback (broken h5py builds can raise
            # H5DSis_scale RuntimeError — see flow_and_diffusion env 2026-08-31)
            try:
                ds.to_netcdf(tmp, encoding=encoding, engine='netcdf4')
            except Exception:
                ds.to_netcdf(tmp, encoding=encoding, engine='h5netcdf')
        os.replace(tmp, str(fp))

    @staticmethod
    def _crop_hk(ds: xr.Dataset) -> xr.Dataset:
        """Crop to the HK forcing box (same convention as data/ifs/ens and
        build_hres_fc0.py): left-closed-right-open lon in [110, 120),
        lat in [15, 25); native coordinate order, no sorting."""
        latn = 'latitude' if 'latitude' in ds.coords else 'lat'
        lonn = 'longitude' if 'longitude' in ds.coords else 'lon'
        lat_m = (ds[latn] >= 15) & (ds[latn] < 25)
        lon_m = (ds[lonn] >= 110) & (ds[lonn] < 120)
        return ds.where(lat_m & lon_m, drop=True)

    def export_to_project_format(
        self,
        init_time_str: str,
        predictions: Optional[xr.Dataset] = None,
        max_lead_h: int = 72,
        num_members: int = 10,
        crop_output_dir: Optional[str] = None,
    ) -> List[str]:
        """Export per-member forecast forcing for member-forced WaveFlow sampling.

        - lead 0h = init-time HRES-fc0 analysis (identical for all members;
          written once as m00 and hard-linked for the rest);
        - 6h..max_lead_h = t0 analysis prepended (time=0) then linearly
          interpolated 12h -> 6h (6h is the midpoint of the 0h analysis and the
          12h forecast - avoids extrapolation NaN);
        - filenames gencast_m{mm}_{init}.nc for mm = 0..num_members-1 (the
          wave_dataset._atm_member_filename convention), layout identical to
          data/ifs/ens;
        - crop_output_dir: also write an HK-cropped mirror tree.
        """
        ts = datetime.strptime(init_time_str, '%Y%m%dT%H')
        year_dir = ts.strftime('%Y')

        if predictions is None:
            pred_path = self.cache_dir / f"gencast_pred_{init_time_str}.nc"
            if not pred_path.exists():
                raise FileNotFoundError(f"Predictions not found: {pred_path}")
            predictions = xr.load_dataset(pred_path)

        preds = predictions[OUTPUT_VARS]
        has_sample = 'sample' in preds[OUTPUT_VARS[0]].dims
        members = list(range(num_members)) if has_sample else [0]

        t0 = self._t0_analysis_dataset(init_time_str)
        for v in OUTPUT_VARS:
            if not np.isfinite(t0[v].values).all():
                raise ValueError(f"Non-finite t0 analysis for {v} @ {init_time_str}")
        if has_sample:
            t0_exp = t0.expand_dims(sample=preds['sample'].values)
        else:
            t0_exp = t0
        t0_exp = t0_exp.expand_dims(time=[np.timedelta64(0, 'ns')])
        full = xr.concat([t0_exp, preds], dim='time')
        preds_6h = self._interpolate_to_6h(full, max_lead_h)

        saved: List[str] = []

        def _save_member(hour: int, m: int, ds_m: xr.Dataset):
            fname = f'gencast_m{m:02d}_{init_time_str}.nc'
            fp = self.output_dir / f'lead_time_{hour}h' / year_dir / fname
            self._atomic_save(ds_m, fp)
            saved.append(str(fp))
            if crop_output_dir is not None:
                fp2 = Path(crop_output_dir) / f'lead_time_{hour}h' / year_dir / fname
                self._atomic_save(self._crop_hk(ds_m), fp2)
                saved.append(str(fp2))

        # lead 0: identical analysis for every member - write m00, hard-link rest
        _save_member(0, 0, t0)
        for m in members[1:]:
            for root in [self.output_dir] + ([Path(crop_output_dir)]
                                             if crop_output_dir is not None else []):
                fp0 = root / 'lead_time_0h' / year_dir / f'gencast_m00_{init_time_str}.nc'
                fpm = root / 'lead_time_0h' / year_dir / f'gencast_m{m:02d}_{init_time_str}.nc'
                try:
                    os.link(fp0, fpm)
                except OSError:
                    self._atomic_save(t0, fpm)
                saved.append(str(fpm))

        # leads 6..72: per-member forecast slices
        for hour in range(6, max_lead_h + 1, 6):
            target_ns = np.timedelta64(hour * 3600 * 10**9, 'ns')
            step = preds_6h.sel(time=target_ns)
            for m in members:
                data_vars = {}
                for v in OUTPUT_VARS:
                    arr = step[v]
                    if has_sample:
                        arr = arr.isel(sample=m)
                    if 'batch' in arr.dims:        # rollout 残留 batch=1 维（lead0
                        arr = arr.isel(batch=0)    # 分析场无此维；统一 squeeze 保多帧一致）
                    arr = arr.astype(np.float32)
                    if not np.isfinite(arr.values).all():
                        raise ValueError(f"Non-finite {v} @ lead {hour}h member {m} "
                                         f"init {init_time_str}")
                    data_vars[v] = arr
                _save_member(hour, m, xr.Dataset(data_vars))

        return saved

    # ------------------------------------------------------------------
    # Step 5: Batch pipeline
    # ------------------------------------------------------------------

    def _member_files_complete(self, its: str, year: str, max_lead_h: int,
                               num_members: int,
                               crop_output_dir: Optional[str]) -> bool:
        """True iff every (lead, member) output file exists in all output trees."""
        roots = [self.output_dir]
        if crop_output_dir is not None:
            roots.append(Path(crop_output_dir))
        for root in roots:
            for lh in range(0, max_lead_h + 1, 6):
                for m in range(num_members):
                    fp = root / f'lead_time_{lh}h' / year / f'gencast_m{m:02d}_{its}.nc'
                    if not fp.exists():
                        return False
        return True

    def run_batch_inference(
        self,
        start_date: str = '20210101',
        end_date: str = '20221231',
        num_ensemble_members: int = 10,
        forecast_steps: int = 6,  # 6 x 12h = 72h
        member_chunk: int = 1,
        steps_per_chunk: int = 1,
        max_lead_h: int = 72,
        skip_existing: bool = True,
        crop_output_dir: Optional[str] = None,
        delete_input_cache: bool = False,
        claim_dir: Optional[str] = None,
        batch_members: bool = False,
        init_plan=None,
    ):
        """End-to-end batch: weights -> input cache -> infer -> per-member export.

        Each init: build the GenCast input cache (HRES-fc0 t-12h/t + ERA5 SST),
        run num_ensemble_members rollouts (kept in memory only - the
        full-variable pred cache is GB-scale and never written in batch mode),
        export per-member forcing files, then optionally delete the input cache
        so disk usage stays at ~1 init transient.

        Args:
            start_date, end_date: Date range 'YYYYMMDD'.
            num_ensemble_members: Ensemble size (figure5 member-forced runs = 10).
            forecast_steps: Number of 12h GenCast steps (6 = 72h).
            max_lead_h: Max lead time in hours for output (72).
            skip_existing: Skip inits whose full member/lead file set exists.
            crop_output_dir: Optional HK-cropped mirror output tree.
            delete_input_cache: Remove gencast_input_*.nc after successful export.
            claim_dir: Optional work-claim directory — 多个 GPU 进程可共享同一
                日期范围;已存在的 init claim 子目录表示他卡在跑,跳过。失败重扫
                前先清掉未完成 init 的 claim。
            init_plan: Optional[List[Tuple[datetime, int, int]]] —— (init,
                forecast_steps, max_lead_h) 列表，覆盖 start/end 全量枚举。
                --select-days 抽样模式用：主 init 全链（6 步到 72h），pad init
                只建 lead 0/6（HK 模型 8 帧历史的 pad 帧只取所在日 init 的
                lead 0/6，见 wave_dataset._build_gencast_frame_plan）。
        """
        # Download model
        self.download_weights()

        # 惰性逐 init 输入缓存（2026-10-15 改）：原先 pre_download_analysis 先把
        # 全部 init 的 2.8GB 输入缓存下完（~4TB，且多进程并发抢写同一 tmp 曾致
        # NetCDF HDF error/段错误）才开始推理；改为循环内领完 claim 后按需构建，
        # 配合 delete_input_cache 用后即删，磁盘只留 ~1 init 瞬态。zarr 句柄只开
        # 一次复用（与 pre_download_analysis 内部同一套句柄）。
        static_vars = self._load_static_vars()
        era5_zarr = xr.open_zarr(ERA5_STATIC_ZARR,
                                 storage_options={"token": "anon"},
                                 consolidated=False)
        hres_zarr = xr.open_zarr(HRES_ANALYSIS_ZARR,
                                 storage_options={"token": "anon"},
                                 consolidated=False)

        # Pre-load model once
        self._load_model()

        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        init_times = []
        d = start
        while d <= end:
            for h in [0, 12]:
                init_times.append(datetime(d.year, d.month, d.day, h))
            d += timedelta(days=1)

        if init_plan is not None:
            plan = sorted(init_plan, key=lambda e: e[0])
        else:
            plan = [(dt, forecast_steps, max_lead_h) for dt in init_times]

        processed, skipped, failed = 0, 0, 0
        if claim_dir is not None:
            os.makedirs(claim_dir, exist_ok=True)  # claim 根目录需先建（2026-10-15：
            # 冷启动时 .claims 不存在，os.mkdir(claim) 报 ENOENT 崩溃循环）
        for dt, fs_plan, mlh in tqdm(plan, desc='GenCast batch'):
            its = dt.strftime('%Y%m%dT%H')
            year = dt.strftime('%Y')

            if skip_existing and self._member_files_complete(
                    its, year, mlh, num_ensemble_members, crop_output_dir):
                skipped += 1
                continue

            if claim_dir is not None:
                claim = Path(claim_dir) / its
                try:
                    os.mkdir(claim)
                except FileExistsError:
                    skipped += 1  # in flight on another GPU process
                    continue

            try:
                if not self._cache_path(dt).exists():
                    self._build_and_cache_input(dt, static_vars, era5_zarr,
                                                hres_zarr, fs_plan)
                preds = self.run_single_inference(
                    dt, num_ensemble_members, fs_plan,
                    cache_predictions=False, member_chunk=member_chunk,
                    batch_members=batch_members,
                    steps_per_chunk=steps_per_chunk)
                preds = preds[OUTPUT_VARS]  # drop 3D vars: keep RAM/export lean
                self.export_to_project_format(
                    its, preds, mlh,
                    num_members=num_ensemble_members,
                    crop_output_dir=crop_output_dir)
                processed += 1
                if delete_input_cache:
                    cp = self._cache_path(dt)
                    if cp.exists():
                        cp.unlink()
            except Exception as e:
                failed += 1
                logger.error(f"Failed {its}: {e}")

        logger.info(f"Done: {processed} processed, {skipped} skipped, "
                    f"{failed} failed, {len(init_times)} total")
        return processed
