#!/usr/bin/env python
"""WaveFlow inference CLI.

    python -m waveflow setup --weights
    python -m waveflow sample --model hk --timestamps 20211013T00 --members 10
    python -m waveflow forecast --init latest --leads 24,48,72
    python -m waveflow plot --run runs/hk_20211013T00

``setup`` downloads the released checkpoints and the shared mesh assets.
``sample`` builds zero-valued wave anchors, fetches the atmospheric forcing from
WeatherBench2 (or from a local NetCDF directory), and runs the released
checkpoint with the paper's sampling recipe. Each member is written to
``<out>/samples/sample_<timestamp>_ens<NNN>.npz`` with the array under the
``sample`` key, shape ``(n_nodes, 4)`` in the order hs, tp, dm, elev.

``forecast`` is the operational path: the forcing comes from the real-time IFS
ensemble forecast on ECMWF open data instead of reanalysis. ``plot`` draws
member / mean / spread maps for a finished run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
HERE = Path(__file__).resolve().parent

MODELS = {
    "hk": {
        "config": "configs/release/hk_4var.yaml",
        "weights": "waveflow-hk-4var-ep49.pt",
        "region": "hk",
        "era5_region": "hk",
        "era5_frames": 8,
        "data_dir": "data/hindcast/case_stubs/processed_npz",
        "batch_size": 32,
        "nodes": 7788,
        "variables": ["hs", "tp", "dm", "elev"],
    },
    "global": {
        "config": "configs/release/global_4var.yaml",
        "weights": "waveflow-global-4var-ep199.pt",
        "region": "global_4var",
        "era5_region": "global",
        "era5_frames": 4,
        "data_dir": "data/global_hindcast/case_stubs/npy_files_4var",
        "batch_size": 2,
        "nodes": 651652,
        "variables": ["hs", "tp", "dm", "elev"],
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def base_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # NFS-safe frame writing
    env.setdefault("TMPDIR", str(REPO / "downloads" / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _stub_region(model: str) -> str:
    return "hk" if model == "hk" else "global_4var"


def make_stubs(model: str, timestamps, out_dir: Path, dry_run: bool) -> None:
    cmd = [sys.executable, str(HERE / "make_case_stubs.py"),
           "--region", _stub_region(model),
           "--timestamps", ",".join(timestamps), "--out", str(out_dir),
           "--no-forcing-check"]
    print("[stubs]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(REPO), env=base_env(), check=True)


def ensure_era5(spec: dict, timestamps, era5_dir: Path | None, dry_run: bool):
    """Return the ``data.era5_dir`` override value, fetching from WB2 if needed."""
    from wb2_era5 import default_out_dir, ensure_wb2_era5

    if era5_dir is None:
        era5_dir = default_out_dir(spec["era5_region"])
    if not dry_run:
        ensure_wb2_era5(timestamps, region=spec["era5_region"],
                        out_dir=era5_dir, frames=spec["era5_frames"])
    return os.path.relpath(era5_dir, REPO)


def ensure_ens(valid: str, leads, members: int, frames: int, tree: str,
               dry_run: bool) -> Path:
    from wb2_ens import GLOBAL_ROOT, HK_ROOT, ensure_wb2_ens

    root = HK_ROOT if tree == "hk" else GLOBAL_ROOT
    for lead in leads:
        if not dry_run:
            ensure_wb2_ens(valid, [lead], frames=frames, members=members,
                           trees=(tree,))
    return root


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_setup(args) -> int:
    import download

    bootstrap.ensure_dirs()
    dl = bootstrap.load_downloads()
    if args.check_env:
        cmd_env(argparse.Namespace())
    download.get_weights(dl)
    print("setup done.")
    return 0


def cmd_env(args) -> int:
    import importlib

    for mod in ("numpy", "scipy", "xarray", "zarr", "gcsfs", "torch", "yaml"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:8s} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:8s} MISSING ({type(exc).__name__})")
    try:
        import torch

        print("  CUDA    ", torch.cuda.is_available(),
              f"({torch.cuda.device_count()} device(s))")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"    GPU{i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    except Exception:  # noqa: BLE001
        print("  CUDA     unavailable")
    print("  weights ", [p.name for p in (REPO / "weights").glob("*.pt")] or "none")
    return 0


def cmd_sample(args) -> int:
    spec = MODELS[args.model]
    timestamps = [t.strip() for t in args.timestamps.split(",") if t.strip()]
    if not timestamps:
        raise SystemExit("give at least one --timestamps value")
    weights = REPO / "weights" / spec["weights"]
    if not weights.exists() and not args.dry_run:
        raise SystemExit(f"missing {weights}; run `python -m waveflow setup --weights`")
    config = REPO / spec["config"]
    workdir = Path(args.out or f"runs/{args.model}_{timestamps[0]}").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO / spec["data_dir"]

    overrides = [
        f"checkpoint.resume_path={weights}",
        "checkpoint.resume_training=false",
        "checkpoint.load_optimizer=false",
        "training.use_ema=false",
        "sampling.use_ema_for_sampling=false",
        f"data.data_dir={data_dir}",
        f"sampling.ensemble_size={args.members}",
        f"sampling.target_timestamps={','.join(timestamps)}",
        f"sampling.sample_split={args.split}",
        f"sampling.batch_size={args.batch_size or spec['batch_size']}",
        f"sampling.steps={args.steps}",
        f"work_dir.path={workdir}",
    ]

    if args.ens_members:
        if not (args.ens_init and args.ens_lead):
            raise SystemExit("--ens-members needs --ens-init and --ens-lead")
        tree = "hk" if args.model == "hk" else "global"
        ens_root = ensure_ens(timestamps[0], [args.ens_lead], args.ens_members,
                              spec["era5_frames"], tree, args.dry_run)
        overrides += [
            "sampling.member_forced=true",
            f"sampling.atm_members={args.ens_members}",
            "sampling.atm_source=ens",
            f"sampling.ens_dir={os.path.relpath(ens_root, REPO)}",
            f"sampling.ens_init={args.ens_init}",
            f"sampling.ens_lead_time={args.ens_lead}",
        ]
        era5_override = os.path.relpath(
            Path(args.era5_dir) if args.era5_dir else
            (REPO / ("data/era5/wb2/" + ("hk" if args.model == "hk" else "global"))),
            REPO)
        overrides += [f"data.era5_dir={era5_override}",
                      "data.era5_file_format=nc",
                      "data.use_preinterpolated_era5=false"]
    else:
        era5_dir = Path(args.era5_dir).resolve() if args.era5_dir else None
        era5_override = ensure_era5(spec, timestamps, era5_dir, args.dry_run)
        overrides += [f"data.era5_dir={era5_override}",
                      "data.era5_file_format=nc",
                      "data.use_preinterpolated_era5=false"]

    make_stubs(args.model, timestamps, data_dir, args.dry_run)

    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={args.nproc}", f"--master_port={free_port()}",
           "main.py", "--config", str(config)]
    for ov in overrides:
        cmd += ["-o", ov]
    print("[sample]", " ".join(cmd))
    if args.dry_run:
        print("[sample] dry run: nothing executed")
        return 0
    subprocess.run(cmd, cwd=str(REPO), env=base_env(), check=True)
    samples = workdir / "samples"
    n = len(list(samples.glob("sample_*.npz")))
    print(f"\n{n} sample file(s) in {samples}")
    print(f"array key: 'sample', shape ({spec['nodes']}, 4), "
          f"columns {spec['variables']}")
    if args.plot:
        from .plot import plot_run

        target = plot_run(workdir, var=args.plot_var)
        print(f"figure: {target}")
    return 0


def _parse_leads(text: str) -> list[int]:
    """``24,48,72`` or ``24-120:24`` -> list of lead hours."""
    leads: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            span, _, step = part.partition(":")
            lo, hi = (int(x) for x in span.split("-"))
            step = int(step or 6)
            leads.extend(range(lo, hi + 1, step))
        else:
            leads.append(int(part))
    return sorted(set(leads))


def cmd_forecast(args) -> int:
    """Real-time IFS ENS -> released checkpoint, one run directory per lead."""
    from datetime import datetime, timedelta

    from . import ifs

    init = args.init
    if init in ("latest", "auto"):
        init = ifs.latest_cycle()
        print(f"[forecast] latest IFS ENS cycle: {init}")
    leads = _parse_leads(args.leads)
    members = list(range(1, args.members + 1))
    out = Path(args.out or f"runs/forecast_{init}").resolve()
    spec = MODELS[args.model]
    weights = REPO / "weights" / spec["weights"]
    if not weights.exists():
        raise SystemExit(f"missing {weights}; run `python -m waveflow setup --weights`")
    print(f"[forecast] {args.model} | init {init} | leads {leads} h | "
          f"{len(members)} ENS members")

    init_dt = datetime.strptime(init, "%Y%m%dT%H")

    if args.model == "hk":
        ens_dir = out / "forcing"
        pairs = sorted({pair for lead in leads for pair in ifs.frame_plan(init, lead)})
        print(f"[forecast] {len(pairs)} unique forcing frames")
        cache = ens_dir / "_index"
        for step, frame_init in pairs:
            target = ens_dir / f"lead_time_{step}h" / frame_init[:4] / f"ifs_ens_{frame_init}.nc"
            if target.exists():
                continue
            if args.dry_run:
                print(f"  would fetch {frame_init}+{step}h")
                continue
            fields = ifs.fetch_mean(frame_init, step, members, cache)
            ifs.write_hk(ens_dir, frame_init, step, fields)
            print(f"  fetched {frame_init}+{step}h")
        data_dir = REPO / "data/hindcast/case_stubs/processed_npz"
        atm = ["sampling.atm_source=ens", "sampling.member_forced=false",
               "sampling.atm_members=1",
               f"sampling.ens_dir={os.path.relpath(ens_dir, REPO)}",
               f"sampling.ens_init={init}"]
    else:
        # global model: IFS ENS -> ERA5-format global npz (4 frames, 6 h apart)
        era5_dir = out / "era5_global"
        data_dir = out / "stubs" / "npy_files_4var"
        steps = sorted({max(0, lead - 6 * i) for lead in leads for i in range(4)})
        frames = sorted({(step, (init_dt + timedelta(hours=step)).strftime("%Y%m%dT%H"))
                         for step in steps})
        print(f"[forecast] {len(frames)} global forcing frames")
        cache = out / "forcing" / "_index"
        for step, valid in frames:
            target = era5_dir / valid[:4] / f"era5_{valid}.npz"
            if target.exists():
                continue
            if args.dry_run:
                print(f"  would fetch {init}+{step}h (valid {valid})")
                continue
            fields = ifs.fetch_mean(init, step, members, cache, crop=False)
            ifs.write_global(era5_dir, valid, fields)
            print(f"  fetched {init}+{step}h -> {target.name}")
        atm = [f"data.era5_dir={os.path.relpath(era5_dir, REPO)}",
               "data.era5_file_format=npz",
               "data.use_preinterpolated_era5=false",
               "sampling.atm_source=era5"]

    for lead in leads:
        valid = (init_dt + timedelta(hours=lead)).strftime("%Y%m%dT%H")
        make_stubs(args.model, [valid], data_dir, args.dry_run)
        workdir = out / f"lead_{lead:03d}h"
        overrides = [
            f"checkpoint.resume_path={weights}",
            "checkpoint.resume_training=false",
            "checkpoint.load_optimizer=false",
            "training.use_ema=false",
            "sampling.use_ema_for_sampling=false",
            f"data.data_dir={data_dir}",
            f"sampling.ensemble_size={args.members}",
            f"sampling.target_timestamps={valid}",
            # a real-time cycle sits outside the 2021-2022 test split
            "sampling.sample_split=train",
            f"data.train_max_year={int(init[:4])}",
            f"sampling.batch_size={spec['batch_size']}",
            f"sampling.steps={args.steps}",
            f"work_dir.path={workdir}",
            *atm,
        ]
        if args.model == "hk":
            overrides.append(f"sampling.ens_lead_time={lead}")
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={args.nproc}", f"--master_port={free_port()}",
               "main.py", "--config", str(REPO / spec["config"])]
        for ov in overrides:
            cmd += ["-o", ov]
        print(f"\n[forecast] lead {lead} h -> valid {valid}")
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO), env=base_env(), check=True)
        if args.plot:
            from .plot import plot_run

            print(f"figure: {plot_run(workdir)}")
    return 0


def cmd_plot(args) -> int:
    from .plot import plot_run

    run = args.run
    if run is None:
        runs = sorted(p for p in (REPO / "runs").glob("*") if p.is_dir())
        if not runs:
            raise SystemExit("no run directory yet; sample first or pass --run")
        run = runs[-1]
        print(f"[plot] using {run}")
    target = plot_run(run, var=args.var, out=args.out, dpi=args.dpi)
    print(f"figure: {target}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m waveflow",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="download the released checkpoints")
    p.add_argument("--weights", action="store_true", help="(default) weights + assets")
    p.add_argument("--check-env", action="store_true", help="report packages first")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("env", help="report packages, CUDA and local weights")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("sample", help="run the released checkpoint on timestamps")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--timestamps", required=True,
                   help="comma-separated valid times, e.g. 20220924T00")
    p.add_argument("--members", type=int, default=10, help="ensemble members")
    p.add_argument("--steps", type=int, default=50, help="ODE steps")
    p.add_argument("--batch-size", type=int, default=0,
                   help="override sampling.batch_size")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None, help="output run directory")
    p.add_argument("--era5-dir", default=None,
                   help="local NetCDF directory instead of WeatherBench2")
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ens-members", type=int, default=0,
                   help="force every member with a member of an IFS ENS run")
    p.add_argument("--ens-init", default=None, help="IFS ENS init time")
    p.add_argument("--ens-lead", type=int, default=None, help="lead time in hours")
    p.add_argument("--plot", action="store_true",
                   help="draw member/mean/std maps when sampling finishes")
    p.add_argument("--plot-var", default="hs", choices=["hs", "tp", "dm", "elev"],
                   help="variable for --plot (default hs)")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("forecast", help="real-time IFS ENS forecast -> samples")
    p.add_argument("--model", choices=sorted(MODELS), default="hk",
                   help="model driven by the real-time IFS ENS forcing")
    p.add_argument("--init", default="latest",
                   help="IFS ENS cycle YYYYMMDDTHH (default: latest on open data)")
    p.add_argument("--leads", default="24,48,72",
                   help="lead hours, e.g. 24,48,72 or 24-120:24")
    p.add_argument("--members", type=int, default=10,
                   help="ENS members to average (1-50, default 10)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("plot", help="draw member/mean/std maps for a run")
    p.add_argument("--run", default=None, help="run directory (default: newest runs/*)")
    p.add_argument("--var", default="hs", choices=["hs", "tp", "dm", "elev"])
    p.add_argument("--out", default=None, help="output PNG (default <run>/figures/)")
    p.add_argument("--dpi", type=int, default=200)
    p.set_defaults(func=cmd_plot)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap.ensure_dirs()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
