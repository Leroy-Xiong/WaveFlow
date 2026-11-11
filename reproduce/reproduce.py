#!/usr/bin/env python
"""WaveFlow v1.0.0 figure-reproduction CLI.

Quick start
-----------
    python reproduce.py setup --download
    python reproduce.py list
    python reproduce.py all
    python reproduce.py verify

Hindcast-derived reference frames are not redistributed; the panels that need
them fail with the exact missing paths (see ``DATA.md``). Redirect them with
``--gt-hk/--gt-global/--gt-global-4var`` or the ``WF_GT_*`` environment
variables.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
PANELS = bootstrap.load_panels()
RAW = REPO / "reproduced" / "_raw"
OUT = REPO / "reproduced"


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def base_env(args=None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    carto = REPO / "data/cartopy_cache"
    carto.mkdir(parents=True, exist_ok=True)
    env["CARTOPY_DATA_DIR"] = str(carto)
    env.setdefault("TMPDIR", str(REPO / "downloads" / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if args is not None:
        for attr, var in (("gt_hk", "WF_GT_HK"), ("gt_global", "WF_GT_GLOBAL"),
                          ("gt_global_4var", "WF_GT_GLOBAL_4VAR")):
            if getattr(args, attr, None):
                env[var] = str(getattr(args, attr))
    return env


def gt_missing(spec) -> list[Path]:
    from check_hindcast import resolve
    return [resolve(p) for p in (spec.get("requires", {}).get("gt") or [])
            if not resolve(p).exists()]


def _print_missing_gt(key, spec, missing):
    print(f"\n[{key}] hindcast reference frame(s) missing:")
    for p in missing:
        print(f"  {p}")
    print("  The hindcast is a third-party product and is not redistributed.")
    print("  See DATA.md; pass --gt-hk/--gt-global/--gt-global-4var or set "
          "WF_GT_* to point at your copy.")


def _panel_cmd(key, spec, args) -> list[str]:
    script = HERE / spec["script"]
    cmd = [sys.executable, str(script), "--out", str(RAW / f"{key}.png"),
           "--dpi", str(spec.get("dpi", 300))]
    if key == "fig3def" and getattr(args, "samples_dir", None):
        cmd += ["--samples-dir", str(args.samples_dir)]
    return cmd


def _whole_cmd(key, spec, args) -> list[str]:
    return [sys.executable, str(HERE / "panels/run_whole.py"),
            "--figure", key, "--out", str(RAW / f"{key}.png")]


def run_panel(key, args, quiet=False) -> bool:
    spec = PANELS[key]
    # GT overrides must be visible to check_hindcast.resolve() before the check
    for attr, var in (("gt_hk", "WF_GT_HK"), ("gt_global", "WF_GT_GLOBAL"),
                      ("gt_global_4var", "WF_GT_GLOBAL_4VAR")):
        if getattr(args, attr, None):
            os.environ[var] = str(getattr(args, attr))
    missing = gt_missing(spec)
    if missing:
        _print_missing_gt(key, spec, missing)
        return False
    env = base_env(args)
    if getattr(args, "samples_dir", None):
        sdir = Path(args.samples_dir)
        if key == "ed5":
            env["WF_HK_SAMPLES"] = str(sdir)
        elif key == "ed6":
            env["WF_GLOBAL_4VAR_SAMPLES"] = str(sdir)
        elif key == "fig6":
            env["WF_FIG6_SAMPLES_M100"] = str(sdir)
    if getattr(args, "q_npz", None):
        env["WF_FIG6_Q_NPZ"] = str(args.q_npz)
    if getattr(args, "q_npz_m10", None):
        env["WF_FIG6_Q_NPZ_M10"] = str(args.q_npz_m10)
    if getattr(args, "summary_csv", None):
        env["WF_FIG6_SUMMARY_CSV"] = str(args.summary_csv)
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    if spec["kind"] == "panel":
        cmd = _panel_cmd(key, spec, args)
    else:
        cmd = _whole_cmd(key, spec, args)
    if args.dry_run:
        print(f"[{key}] {' '.join(cmd)}")
        print(f"[{key}] gt: {spec['requires'].get('gt') or 'none'}")
        return True
    if not quiet:
        print(f"\n=== [{key}] {spec['title']}")
        print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)

    if spec["kind"] == "panel":
        import crop_panels
        written = crop_panels.crop_outputs(PANELS, RAW, OUT, only={key})
        return bool(written)
    src = RAW / f"{key}.png"
    dst = OUT / spec["copy_to"]
    if not src.exists():
        raise SystemExit(f"{key}: expected output {src} missing")
    shutil.copy2(src, dst)
    print(f"  -> {dst}")
    return True


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_setup(args):
    rc = 0
    if args.check_env:
        rc = subprocess.call([sys.executable, str(HERE / "check_env.py")] +
                             (["--infer"] if args.infer else []))
    if args.download or args.weights or args.figure_data:
        cmd = [sys.executable, str(HERE / "download.py")]
        if args.download or (args.weights and args.figure_data):
            cmd.append("--all")
        elif args.weights:
            cmd.append("--weights")
        elif args.figure_data:
            cmd.append("--figure-data")
        rc |= subprocess.call(cmd)
    bootstrap.ensure_dirs()
    gt = subprocess.call([sys.executable, str(HERE / "check_hindcast.py")])
    if gt and not args.download:
        print("\nNote: some reference-bearing panels need a hindcast frame; "
              "the rest can be reproduced now.")
    print("\nsetup done. next:  python reproduce.py all")
    return rc


def cmd_list(args):
    print(f"{'id':10s} {'kind':6s} {'needs GT':9s} title")
    for key, spec in PANELS.items():
        n_gt = len(spec.get("requires", {}).get("gt") or [])
        print(f"{key:10s} {spec['kind']:6s} {('yes' if n_gt else 'no'):9s} "
              f"{spec['title']}")


def cmd_all(args):
    ok, skipped = [], []
    for key in PANELS:
        try:
            done = run_panel(key, args, quiet=args.quiet)
        except subprocess.CalledProcessError as exc:
            print(f"[{key}] FAILED (exit {exc.returncode})")
            if args.strict:
                return 1
            continue
        if done:
            ok.append(key)
        else:
            skipped.append(key)
    print("\nreproduced:", ", ".join(ok) or "-")
    if skipped:
        print("skipped (missing hindcast):", ", ".join(skipped))
    if args.verify:
        return cmd_verify(argparse.Namespace(reproduced="reproduced",
                                             reference="reference",
                                             json_out="reproduced/verification_report.json",
                                             mean_tol=1.0, frac_tol=0.001))
    if args.strict and skipped:
        return 1
    return 0


def cmd_verify(args):
    cmd = [sys.executable, str(HERE / "verify.py"),
           "--reproduced", str(args.reproduced),
           "--reference", str(args.reference),
           "--json-out", str(args.json_out),
           "--mean-tol", str(args.mean_tol),
           "--frac-tol", str(args.frac_tol)]
    return subprocess.call(cmd, cwd=str(REPO))


def cmd_check_hindcast(args):
    cmd = [sys.executable, str(HERE / "check_hindcast.py")]
    for attr, flag in (("gt_hk", "--gt-hk"), ("gt_global", "--gt-global"),
                       ("gt_global_4var", "--gt-global-4var")):
        if getattr(args, attr, None):
            cmd += [flag, str(getattr(args, attr))]
    return subprocess.call(cmd, cwd=str(REPO))


def _expand_timestamps(spec) -> list[str]:
    if spec.get("timestamps"):
        return list(spec["timestamps"])
    rng = spec.get("timestamps_range")
    if not rng:
        raise SystemExit("infer block needs timestamps/timestamps_range")
    from datetime import datetime, timedelta
    t = datetime.strptime(rng["start"], "%Y%m%dT%H")
    stop = datetime.strptime(rng["end"], "%Y%m%dT%H")
    step = int(rng.get("step_hours", 1))
    out = []
    while t <= stop:
        out.append(t.strftime("%Y%m%dT%H"))
        t += timedelta(hours=step)
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def cmd_infer(args):
    key = {"fig3f": "fig3def"}.get(args.panel, args.panel)
    spec = PANELS[key]
    inf = spec.get("infer")
    if not inf:
        raise SystemExit(f"{key}: no inference recipe in panels.yaml "
                         "(render tier only; see REPRODUCE.md for manual steps)")
    timestamps = _expand_timestamps(inf)
    if args.timestamps:
        timestamps = [t.strip() for t in args.timestamps.split(",") if t.strip()]
    members = int(args.members or inf.get("members", 10))
    workdir = REPO / (args.out or inf.get("workdir", f"output/repro/{key}"))
    data_dir = REPO / inf["data_dir"]
    region = inf["region"]
    # 1) zero-valued wave anchors (timestamp enumeration only)
    stub_cmd = [sys.executable, str(HERE / "make_case_stubs.py"),
                "--region", region, "--timestamps", ",".join(timestamps),
                "--out", str(data_dir)]
    print("[infer] stubs:", " ".join(stub_cmd))
    subprocess.run(stub_cmd, cwd=str(REPO), env=base_env(args), check=True)
    # 2) sampling
    weights = REPO / inf["weights"]
    if not weights.exists():
        raise SystemExit(f"missing weights {weights}; run "
                         "`python reproduce.py setup --weights`")
    config = REPO / inf["config"]
    port = _free_port()
    nproc = int(args.nproc or 1)
    workdir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"checkpoint.resume_path={weights}",
        "checkpoint.resume_training=false",
        "checkpoint.load_optimizer=false",
        "training.use_ema=false",
        "sampling.use_ema_for_sampling=false",
        f"data.data_dir={data_dir}",
        f"sampling.ensemble_size={members}",
        f"sampling.target_timestamps={','.join(timestamps)}",
        f"sampling.sample_split={inf.get('sample_split', 'test')}",
        f"sampling.atm_source={inf.get('atm_source', 'era5')}",
        f"sampling.batch_size={inf.get('batch_size', 16)}",
        f"sampling.steps={inf.get('steps', 50)}",
        f"work_dir.path={workdir}",
    ]
    if inf.get("train_max_year"):
        overrides.append(f"data.train_max_year={inf['train_max_year']}")
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={nproc}", f"--master_port={port}",
           "main.py", "--config", str(config), "--mode", "sample"]
    for ov in overrides:
        cmd += ["-o", ov]
    print("[infer] sampling:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=str(REPO), env=base_env(args), check=True)
    samples = workdir / "samples"
    if not args.dry_run:
        n = len(list(samples.glob("sample_*.npz")))
        print(f"[infer] {n} sample files in {samples}")
    # 3) optional overtopping step (Fig. 6)
    render_env = {}
    if inf.get("overtopping"):
        ov = inf["overtopping"]
        # compute_overtopping_saola.py expects
        # <run>/case_ragasa2025_m100/test/samples; alias the flat sampling dir
        case_dir = workdir / "case_ragasa2025_m100" / "test"
        case_dir.mkdir(parents=True, exist_ok=True)
        link = case_dir / "samples"
        if not link.exists():
            rel = os.path.relpath(samples, case_dir)
            os.symlink(rel, link)
        ov_out = REPO / ov["out_dir"]
        ov_out.mkdir(parents=True, exist_ok=True)
        env = base_env(args)
        env["WF_HK_RUN"] = str(workdir)
        env["WF_FIG6_OVERTOPPING_DIR"] = str(ov_out)
        ov_cmd = [sys.executable, str(REPO / ov["script"])] + \
                 [str(a) for a in ov.get("args", [])]
        print("[infer] overtopping:", " ".join(ov_cmd))
        if not args.dry_run:
            subprocess.run(ov_cmd, cwd=str(REPO), env=env, check=True)
        names = sorted(ov_out.glob("ragasa_q_m100_*.npz"))
        if names:
            render_env[ov.get("q_env", "WF_FIG6_Q_NPZ")] = str(names[0])
        names10 = sorted(ov_out.glob("ragasa_q_m10_*.npz"))
        if names10:
            render_env[ov.get("q_m10_env", "WF_FIG6_Q_NPZ_M10")] = str(names10[0])
        # Panel c (M=10 vs M=100 exposure) keeps the shipped aggregate summary;
        # the re-inference pass refreshes the M=100 q products used by b/d.
        summaries = sorted(ov_out.glob("ragasa_segment_summary_*.csv"))
        if summaries and args.which_summary:
            render_env[ov.get("summary_env", "WF_FIG6_SUMMARY_CSV")] = str(summaries[0])
    # 4) render from the new samples
    if args.render_only or not args.no_render:
        if inf.get("samples_env"):
            render_env[inf["samples_env"]] = str(samples)
        if inf.get("samples_arg") == "--samples-dir":
            args.samples_dir = str(samples)
        else:
            args.samples_dir = None
        for k, v in render_env.items():
            os.environ[k] = v
        return 0 if run_panel(key, args) else 1
    return 0


def cmd_verify_args(args):
    return cmd_verify(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--gt-hk", default=None)
        p.add_argument("--gt-global", default=None)
        p.add_argument("--gt-global-4var", default=None)
        p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("setup", help="check env, download and extract assets")
    p.add_argument("--check-env", action="store_true", default=True)
    p.add_argument("--infer", action="store_true", help="also check torch/CUDA")
    p.add_argument("--download", action="store_true")
    p.add_argument("--weights", action="store_true")
    p.add_argument("--figure-data", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("list", help="list reproducible panels")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("check-hindcast", help="check the required hindcast frames")
    add_common(p)
    p.set_defaults(func=cmd_check_hindcast)

    p = sub.add_parser("verify", help="compare reproduced panels to reference")
    p.add_argument("--reproduced", default="reproduced")
    p.add_argument("--reference", default="reference")
    p.add_argument("--json-out", default="reproduced/verification_report.json")
    p.add_argument("--mean-tol", type=float, default=1.0)
    p.add_argument("--frac-tol", type=float, default=0.001)
    p.set_defaults(func=cmd_verify_args)

    p = sub.add_parser("all", help="run every panel")
    add_common(p)
    p.add_argument("--strict", action="store_true",
                   help="non-zero exit if a hindcast-dependent panel is skipped")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--samples-dir", default=None)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("infer", help="re-run inference with the released weights")
    infer_keys = [k for k, v in PANELS.items() if v.get("infer")]
    if "fig3def" in infer_keys:
        infer_keys.append("fig3f")  # documented alias for the Fig. 3f recipe
    p.add_argument("panel", choices=infer_keys)
    add_common(p)
    p.add_argument("--members", type=int, default=None)
    p.add_argument("--timestamps", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--which-summary", action="store_true",
                   help="also use the freshly computed exposure summary (needs M=10)")
    p.set_defaults(func=cmd_infer)

    for key, spec in PANELS.items():
        p = sub.add_parser(key, help=spec["title"])
        add_common(p)
        p.add_argument("--samples-dir", default=None,
                       help="render from these samples instead of the shipped ones")
        p.add_argument("--q-npz", default=None, help="Fig. 6 q m100 npz override")
        p.add_argument("--q-npz-m10", default=None)
        p.add_argument("--summary-csv", default=None)
        p.set_defaults(func=lambda a, k=key: 0 if run_panel(k, a) else 1)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    bootstrap.ensure_dirs()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
