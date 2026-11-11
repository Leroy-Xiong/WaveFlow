#!/usr/bin/env python
"""Compare reproduced panels against the reference panels shipped in
``reference/`` and write a JSON/Markdown report.

The reference panels are crops of the paper figures, produced with the same
millimetre boxes as the reproduced outputs, so a correct reproduction is
pixel-close (tiny antialiasing differences may remain). Stochastic re-inference
is expected to differ; use ``reproduce.py verify`` on render-tier outputs.

    python verify.py --reproduced reproduced --reference reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    x = a.mean(axis=2).astype(np.float64)
    y = b.mean(axis=2).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x = uniform_filter(x, 7)
    mu_y = uniform_filter(y, 7)
    sx = uniform_filter(x * x, 7) - mu_x ** 2
    sy = uniform_filter(y * y, 7) - mu_y ** 2
    sxy = uniform_filter(x * y, 7) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2)
    return float(np.mean(num / den))


def compare(ref_path: Path, out_path: Path):
    ref = Image.open(ref_path).convert("RGB")
    out = Image.open(out_path).convert("RGB")
    resized = False
    if ref.size != out.size:
        ref = ref.resize(out.size, Image.LANCZOS)
        resized = True
    a = np.asarray(ref, dtype=np.float32)
    b = np.asarray(out, dtype=np.float32)
    diff = np.abs(a - b)
    return {
        "reference": str(ref_path),
        "output": str(out_path),
        "shape": list(out.size),
        "reference_resized": resized,
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "max_abs": float(diff.max()),
        "frac_gt8": float((diff > 8.0).mean()),
        "ssim": ssim_gray(a, b),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduced", default="reproduced")
    ap.add_argument("--reference", default="reference")
    ap.add_argument("--json-out", default="reproduced/verification_report.json")
    ap.add_argument("--mean-tol", type=float, default=1.0,
                    help="max mean absolute difference (0-255 scale)")
    ap.add_argument("--frac-tol", type=float, default=0.001,
                    help="max fraction of pixels differing by more than 8/255")
    args = ap.parse_args(argv)

    repro = Path(args.reproduced)
    refdir = Path(args.reference)
    refs = sorted(refdir.glob("*.png"))
    if not refs:
        raise SystemExit(f"no reference PNGs in {refdir}")
    report, failures = [], []
    print(f"{'panel':10s} {'mean':>6s} {'p99':>6s} {'max':>6s} "
          f"{'>8px':>7s} {'SSIM':>7s}  status")
    for ref in refs:
        out = repro / ref.name
        if not out.exists():
            report.append({"panel": ref.stem, "status": "missing"})
            failures.append(ref.stem)
            print(f"{ref.stem:10s} {'-':>6s} {'-':>6s} {'-':>6s} {'-':>7s} "
                  f"{'-':>7s}  MISSING")
            continue
        r = compare(ref, out)
        ok = (r["mean_abs"] <= args.mean_tol and r["frac_gt8"] <= args.frac_tol)
        r.update({"panel": ref.stem, "status": "pass" if ok else "fail"})
        report.append(r)
        if not ok:
            failures.append(ref.stem)
        print(f"{ref.stem:10s} {r['mean_abs']:6.2f} {r['p99_abs']:6.2f} "
              f"{r['max_abs']:6.1f} {100 * r['frac_gt8']:6.2f}% {r['ssim']:7.4f}"
              f"  {'pass' if ok else 'FAIL'}")
    payload = {
        "mean_tolerance": args.mean_tol,
        "frac_tolerance": args.frac_tol,
        "panels": report,
        "failures": failures,
        "result": "PASS" if not failures else "FAIL",
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nreport: {out}")
    print("RESULT:", payload["result"])
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
