#!/usr/bin/env python
"""Grid of pairwise joint densities: hindcast against model, Hong Kong and global.

Rows are variable pairs, columns are (Hong Kong hindcast, Hong Kong model,
global hindcast, global model).  Each density is normalised to the largest
cell of its (pair, domain) group, so the hindcast and the model of a domain
share one colour scale.

Layout rules used here (same as the other Extended Data figures):
  * one axis label per row and domain, centred under its two panels, instead
    of 24 repeated labels;
  * plain-text axis labels (``Hs``, ``Dm``) rather than mathtext subscripts,
    which shrink below the 5 pt floor at this figure width;
  * every glyph at 5 pt or more, editable text in the PDF/SVG exports.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)
OUT_DIR = os.path.join(ROOT, "output/plot/figure_ed_joint")
OUT_BASE = os.path.join(OUT_DIR, "figure_ed_pairs")

FW, FH = 180.0, 196.0
MM = 1 / 25.4
VMIN = 1.5e-4

PAIRS = [("hs", "tp"), ("hs", "elev"), ("tp", "elev"), ("hs", "dm"),
         ("tp", "dm"), ("dm", "elev")]
AX_LABEL = {"hs": "Hs (m)", "tp": "period (s)",
            "elev": "water level (m)", "dm": "Dm (\u00b0)"}
RANGES = {
    "hk": {"hs": (0.0, 3.0), "tp": (0.0, 13.0),
           "elev": (-1.5, 1.5), "dm": (0.0, 360.0)},
    "global": {"hs": (0.0, 12.0), "tp": (0.0, 16.0),
               "elev": (-1.5, 1.5), "dm": (0.0, 360.0)},
}

FS_LABEL = 6.4
FS_TICK = 5.6
FS_TITLE = 6.6
FS_CB = 5.2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                        "DejaVu Sans"],
    "font.size": 6.4, "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": None,
})


def load(domain):
    f = os.path.join(OUT_DIR, f"pairs_full_{domain}.npz")
    if not os.path.exists(f):
        print(f"[warn] {f} missing")
        return None
    d = np.load(f)
    H_gt, H_mo = d["H_gt"], d["H_mo"]
    fe = os.path.join(OUT_DIR, f"pairs_full_{domain}_extra.npz")
    if os.path.exists(fe):
        e = np.load(fe)
        H_gt = np.concatenate([H_gt, e["H_gt"]], axis=0)
        H_mo = np.concatenate([H_mo, e["H_mo"]], axis=0)
    return dict(H_gt=H_gt, H_mo=H_mo, n_ts=int(d["n_ts"]))


def plot_panel(ax, H, xr, yr, vmax):
    H = gaussian_filter(H, 0.8, mode="constant")
    rel = H / max(vmax, 1e-9)
    rel = np.where(rel < VMIN, np.nan, rel)
    nb_x, nb_y = H.shape
    xe = np.linspace(*xr, nb_x + 1)
    ye = np.linspace(*yr, nb_y + 1)
    ax.pcolormesh(xe, ye, rel.T, cmap="viridis", norm=LogNorm(VMIN, 1.0),
                  shading="flat")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# final layout in mm
COLS = (14.0, 55.0, 96.0, 137.0)
ROWS = (13.0, 42.0, 71.0, 100.0, 129.0, 158.0)
PW, PH = 37.0, 22.0


def build_figure(data):
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    axes = {}

    for r, (a, b) in enumerate(PAIRS):
        y0 = ROWS[r]
        for dom_i, dom in enumerate(("hk", "global")):
            if data[dom] is None:
                continue
            Hi = PAIRS.index((a, b))
            H_g = data[dom]["H_gt"][Hi]
            H_m = data[dom]["H_mo"][Hi]
            # probability per cell first: the model has ten times the samples
            H_g = H_g / max(H_g.sum(), 1e-12)
            H_m = H_m / max(H_m.sum(), 1e-12)
            vmax = max(H_g.max(), H_m.max())
            for k, H in enumerate((H_g, H_m)):
                c = 2 * dom_i + k
                ax = fig.add_axes([COLS[c] / FW, 1 - (y0 + PH) / FH,
                                   PW / FW, PH / FH])
                axes[(r, c)] = ax
                plot_panel(ax, H, RANGES[dom][a], RANGES[dom][b], vmax)
                if c == 0:
                    ax.set_ylabel(AX_LABEL[b], labelpad=1.5)
                else:
                    ax.tick_params(labelleft=False)
        # one centred x label per row and domain instead of four repeats
        for dom_i in (0, 1):
            x_c = (COLS[2 * dom_i] + COLS[2 * dom_i + 1] + PW) / 2
            fig.text(x_c / FW, 1 - (y0 + PH + 5.4) / FH, AX_LABEL[a],
                     ha="center", va="center", fontsize=FS_LABEL)

    names = ("Reference (Hong Kong)", "WaveFlow (Hong Kong)",
             "Reference (Global)", "WaveFlow (Global)")
    for c, nm in enumerate(names):
        fig.text((COLS[c] + PW / 2) / FW, 1 - 8.0 / FH, nm, ha="center",
                 fontsize=FS_TITLE, fontweight="bold")

    # shared colour ramp (the scale is normalised per pair and domain)
    from matplotlib.cm import ScalarMappable
    cax = fig.add_axes([COLS[0] / FW, 1 - 189.0 / FH, 46.0 / FW, 1.6 / FH])
    sm = ScalarMappable(norm=LogNorm(VMIN, 1.0), cmap="viridis")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.minorticks_off()
    cb.set_ticks([1e-3, 1e-2, 1e-1, 1.0])
    # plain decimals: the default 10**-3 formatter renders its exponent at
    # 0.7x, i.e. 3.6 pt at this tick size, below the 5 pt glyph floor
    cb.set_ticklabels(["0.001", "0.01", "0.1", "1"])
    cb.ax.tick_params(labelsize=FS_CB, length=1.4, pad=1.2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("relative probability density (log, per pair and domain)",
                 fontsize=FS_CB, labelpad=1.2)
    return fig, axes


def run_alignment_gate(fig, axes, base):
    """Optional render-time gate; enabled by pointing NATURE_QA_SCRIPTS at the
    panel-alignment auditor directory."""
    scripts = os.environ.get("NATURE_QA_SCRIPTS")
    if not scripts or not axes:
        return
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    ids = {(r, c): f"r{r}c{c}" for (r, c) in axes}
    require_matplotlib_panel_alignment(
        fig,
        axes=[axes[k] for k in sorted(axes)],
        panel_ids=[ids[k] for k in sorted(axes)],
        row_groups=[[ids[(r, c)] for c in range(4) if (r, c) in ids]
                    for r in range(len(ROWS))],
        column_groups=[[ids[(r, c)] for r in range(len(ROWS)) if (r, c) in ids]
                       for c in range(4)],
        json_out=base + ".alignment.json",
        overlay_svg=base + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )


def main():
    data = {d: load(d) for d in ("hk", "global")}
    fig, axes = build_figure(data)
    run_alignment_gate(fig, axes, OUT_BASE)
    fig.savefig(OUT_BASE + ".png", dpi=600)
    fig.savefig(OUT_BASE + ".pdf")
    # no svg here: 24 pcolormesh panels would be ~13 MB of vector quads, and
    # the pdf already carries editable text (fonttype 42)
    plt.close(fig)
    print("saved", OUT_BASE)


if __name__ == "__main__":
    main()
