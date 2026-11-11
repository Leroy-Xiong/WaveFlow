#!/usr/bin/env python
"""Run one of the complete paper figure scripts (Fig. 6, ED 3, ED 5, ED 6).

These figures are reproduced by running the original, unmodified paper script;
only environment variables (case switch, hindcast/sample roots) are set.

    python run_whole.py --figure fig6 --out reproduced/_raw/fig6.png
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bootstrap  # noqa: E402

REPO = bootstrap.REPO
PANELS = bootstrap.load_panels()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", required=True,
                    choices=["fig6", "ed3", "ed5", "ed6"])
    ap.add_argument("--out", default=None,
                    help="copy the rendered PNG here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    spec = PANELS[args.figure]
    script = REPO / spec["script"]
    output = REPO / spec["output"]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
    if args.figure == "fig6":
        env.setdefault("FIG6_SKIP_PDF", "1")  # 40 MB vector PDF not needed
    if args.figure == "ed6":
        env.setdefault("ED6_SKIP_PDF", "1")  # ~300 MB vector PDF not needed
    print(f"[{args.figure}] {' '.join([sys.executable, str(script)])}")
    if args.dry_run:
        print(f"  env: {spec.get('env') or {}}")
        print(f"  expected output: {output}")
        return 0
    before = output.stat().st_mtime if output.exists() else None
    subprocess.run([sys.executable, str(script)], cwd=str(REPO), env=env,
                   check=True)
    if not output.exists():
        raise SystemExit(f"{args.figure}: script finished but {output} is missing")
    if before is not None and output.stat().st_mtime == before:
        print(f"  warning: {output} was not rewritten (cached?)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, out)
        print(f"[{args.figure}] copied {output} -> {out}")
    else:
        print(f"[{args.figure}] output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
