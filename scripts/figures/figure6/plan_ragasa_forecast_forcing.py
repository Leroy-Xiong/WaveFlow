#!/usr/bin/env python
"""Build the IFS-ENS forcing download plan for one forecast issue time.

Fig. 6 (Ragasa) panel d/e add one operational-style forecast: a SINGLE issue
time (default 2025-09-22 12Z) whose 3-hourly valid times cover the analysis
window 2025-09-23T00Z .. 2025-09-25T00Z (lead 12-60 h). Feeding one issue time
means every valid time shares the same main-chain init while the lead varies;
the 8-frame input history of each valid time also pulls the *pad* frames from
earlier 00/12Z cycles at lead 0-9 h (3-hourly valid times make the pad land on
non-zero leads, so step 0 alone is not enough).

The frame plan itself comes from modules/plot/compare_era5_ens, which mirrors
WaveDataset._build_ens_frame_plan, so the downloaded tree matches exactly what
the sampler will ask for.

Usage:
    python scripts/figures/figure6/plan_ragasa_forecast_forcing.py \
        --init 20250922T12 --valid 20250923T00..20250925T00:3
    # -> output/plot/figure6/ragasa_forecast_forcing_plan.json
    #    then: python scripts/live/cli.py fetch --plan-json <that json> \
    #          --members 1-10 --per-member --out data/ifs/ens_members_hk_ragasa
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_DIR)))
os.chdir(_REPO_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modules.plot.compare_era5_ens import _build_ens_frame_plan  # noqa: E402


def parse_valid(text):
    """'A..B:step' / 'A,B' -> list of 'YYYYMMDDTHH' (step in hours, default 6)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            rng, _, stride = part.partition(":")
            start, stop = (datetime.strptime(s, "%Y%m%dT%H")
                           for s in rng.split(".."))
            step = int(stride) if stride else 6
            t = start
            while t <= stop:
                out.append(t.strftime("%Y%m%dT%H"))
                t += timedelta(hours=step)
        else:
            out.append(part)
    return out


def build_plan(init, valid_times, num_frames=8):
    """{init: sorted steps} covering every frame the model consumes."""
    plan: dict[str, set[int]] = {}
    for ts in valid_times:
        for frame in _build_ens_frame_plan(ts, 0, num_frames, ens_init=init):
            lead = int(frame["lead_time"])
            if lead < 0:
                raise SystemExit(f"{ts}: negative lead {lead} for init {init}")
            plan.setdefault(frame["init_time"], set()).add(lead)
    return {k: sorted(v) for k, v in sorted(plan.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="20250922T12",
                    help="forecast issue time (default 20250922T12)")
    ap.add_argument("--valid", default="20250923T00..20250925T00:3",
                    help="target valid times, 'A..B:step' or comma list "
                         "(default 20250923T00..20250925T00:3)")
    ap.add_argument("--frames", type=int, default=8,
                    help="atmospheric input frames per sample (default 8)")
    ap.add_argument("--members", type=int, default=10,
                    help="members per cycle, only used for the size estimate")
    ap.add_argument("--out",
                    default="output/plot/figure6/ragasa_forecast_forcing_plan.json",
                    help="output plan JSON")
    args = ap.parse_args(argv)

    valid = parse_valid(args.valid)
    if not valid:
        raise SystemExit("no valid times parsed from --valid")
    plan = build_plan(args.init, valid, args.frames)

    n_pairs = sum(len(v) for v in plan.values())
    est_gb = n_pairs * 3 * args.members * 1.3e6 / 1e9
    lead0 = int((datetime.strptime(valid[0], "%Y%m%dT%H")
                 - datetime.strptime(args.init, "%Y%m%dT%H")).total_seconds() // 3600)
    lead1 = int((datetime.strptime(valid[-1], "%Y%m%dT%H")
                 - datetime.strptime(args.init, "%Y%m%dT%H")).total_seconds() // 3600)
    print(f"issue time {args.init}; {len(valid)} valid times "
          f"({valid[0]} .. {valid[-1]}, lead {lead0}-{lead1} h)")
    print(f"frame plan: {n_pairs} (init, step) pairs from {len(plan)} inits, "
          f"{args.members} members -> {n_pairs * 3 * args.members} fields "
          f"(~{est_gb:.1f} GB)")
    for init in sorted(plan):
        leads = plan[init]
        print(f"  {init}  n={len(leads):2d}  steps {leads}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(plan, fh, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
