#!/usr/bin/env python
"""Check the Python environment required by the figure-reproduction package.

Render-only reproduction needs the plotting/IO stack; weight inference
additionally needs torch + CUDA. Missing *optional* pieces are reported without
failing unless they are needed by the requested tier.

    python check_env.py
    python check_env.py --infer
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO

CRITICAL = [
    ("yaml", "pyyaml"), ("numpy", "numpy"), ("scipy", "scipy"),
    ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("PIL", "pillow"),
    ("cartopy", "cartopy"), ("shapely", "shapely"), ("fitz", "pymupdf"),
]
RENDER_EXTRA = [
    ("xarray", "xarray"), ("rasterio", "rasterio"), ("pyproj", "pyproj"),
    ("affine", "affine"), ("netCDF4", "netcdf4"), ("h5py", "h5py"),
    ("pyarrow", "pyarrow"),
]
INFER_EXTRA = [("torch", "torch")]
WB2_EXTRA = [
    ("xarray", "xarray"), ("zarr", "zarr"), ("gcsfs", "gcsfs"),
    ("h5netcdf", "h5netcdf"),
]


def _version(modname: str, distname: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(distname)
    except Exception:
        try:
            mod = __import__(modname)
            return str(getattr(mod, "__version__", "?"))
        except Exception:
            return "?"


def check(mods):
    missing = []
    for mod, dist in mods:
        if importlib.util.find_spec(mod) is None:
            missing.append((mod, dist))
        else:
            print(f"  OK   {mod:14s} {_version(mod, dist)}")
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer", action="store_true",
                    help="also require torch + CUDA (weight inference)")
    args = ap.parse_args(argv)

    ok = True
    print(f"python      {sys.version.split()[0]}  ({platform.platform()})")
    print(f"repository  {REPO}")
    print("\n[critical packages]")
    missing = check(CRITICAL)
    if missing:
        ok = False
        print("  MISSING:", ", ".join(f"{m} ({d})" for m, d in missing))
    print("\n[render extras]")
    missing = check(RENDER_EXTRA)
    if missing:
        print("  note: missing optional render packages:",
              ", ".join(f"{m} ({d})" for m, d in missing))

    if args.infer:
        print("\n[inference]")
        missing = check(INFER_EXTRA)
        if missing:
            ok = False
            print("  MISSING:", ", ".join(f"{m} ({d})" for m, d in missing))
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"  OK   CUDA {torch.version.cuda}  "
                          f"devices: {torch.cuda.device_count()} x "
                          f"{torch.cuda.get_device_name(0)}")
                    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"  GPU memory: {mem:.1f} GB")
                    if mem < 40:
                        print("  warning: HK inference may fit, global models need "
                              "~80 GB (see REPRODUCE.md)")
                else:
                    print("  WARNING: CUDA is not available; inference will be "
                          "refused (CPU sampling does not reproduce the GPU result)")
            except Exception as exc:
                print(f"  WARNING: torch check failed: {exc}")

        print("\n[WB2 ERA5 download]")
        missing = check(WB2_EXTRA)
        if missing:
            print("  note: missing WB2 packages; Fig. 3f cannot download ERA5:",
                  ", ".join(f"{m} ({d})" for m, d in missing))

    print("\n[directories]")
    for name in ("weights", "downloads", "reproduced"):
        p = REPO / name
        print(f"  {'OK ' if p.exists() else '-- '} {p}")

    print("\n[fonts]")
    font_dir = REPO / "scripts/figures/fonts"
    fonts = sorted(font_dir.glob("texgyreheros-*.otf"))
    print(f"  bundled TeX Gyre Heros: {len(fonts)}/4 in {font_dir}")
    if len(fonts) != 4:
        print("  WARNING: fonts missing; the figure text will not match the paper")

    print("\n[optional]")
    for exe in ("kpsewhich", "conda"):
        p = shutil.which(exe)
        print(f"  {exe:10s} {p or '-'}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
