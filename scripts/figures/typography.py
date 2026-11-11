#!/usr/bin/env python
"""Nature 风格的字体设置（Helvetica 系）。

问题：默认的 matplotlib 会退回 **DejaVu Sans**（字形更宽、笔画更圆、`$H_s$`
等数学符号还走另一套 DejaVu mathtext），和 Nature 正文用的 Helvetica / Arial
差别明显，"一眼就不是 Nature"。

做法：
1. 优先用系统里已有的 Helvetica 系字体（Arial / Helvetica / Liberation Sans /
   Nimbus Sans / Arimo），本机没有就把 TeX Live 自带的 **TeX Gyre Heros**
   （Helvetica 的度量兼容克隆，GPL+font exception，可随图嵌入 PDF）注册进来；
2. 把 mathtext（`$H_s$`、`$\\tau$`、`$P(H_s>4$ m$)$` 等）也绑到同一个字面，
   正文字体与公式字形同源——这一步是"看起来像 Nature"的关键；
3. 保持 pdf.fonttype=42（TrueType 嵌入，投稿系统要求）。

用法（在设置 rcParams 之前调用一次）：
    import typography as TY
    TY.use_nature_fonts()          # 返回实际选中的字体名
"""
import glob
import os
import shutil
import subprocess

import matplotlib as mpl
from matplotlib import font_manager as fm

__all__ = ["use_nature_fonts", "enforce_glyph_floor", "FAMILY",
           "GLYPH_FLOOR_PT", "MATH_MIN_PT"]

# 按 Nature 排版习惯的优先级（Helvetica / 其度量兼容替代）
_PREFERRED = ("Arial", "Helvetica", "TeX Gyre Heros", "Liberation Sans",
              "Nimbus Sans", "Arimo", "DejaVu Sans")
FAMILY = None

# --- 5 pt 字形底线 ---------------------------------------------------------
# 投稿规定"每个落到 PDF 里的字形都 >= 5 pt"，而 mathtext 的上下标固定按父字号
# 的 0.7 倍缩（matplotlib 内部常量，没有 rcParam 可调）。于是 7 pt 的 $H_s$
# 会印出 4.9 pt 的 s，6 pt 的会印出 4.2 pt —— 都破底线。
# 要拿到 >= 5 pt 的上下标，父字号至少 5/0.7 = 7.14 pt，这里统一取 7.2 pt
# （与 7 pt 档肉眼无差，不产生新的一级字号）。
GLYPH_FLOOR_PT = 5.0
MATHTEXT_SHRINK = 0.7
MATH_MIN_PT = 7.2  # = 5 pt / 0.7, rounded up


def _installed_families():
    return {f.name for f in fm.fontManager.ttflist}


def _kpsewhich_binaries():
    cand = [shutil.which("kpsewhich")]
    for pat in (os.path.expanduser("~/texlive/*/bin/*/kpsewhich"),
                "/usr/local/texlive/*/bin/*/kpsewhich",
                "/usr/share/texlive/bin/*/kpsewhich"):
        cand.extend(sorted(glob.glob(pat), reverse=True))
    return [c for c in cand if c and os.path.exists(c)]


def _texgyre_heros_files():
    """定位 TeX Gyre Heros 的 4 个 OTF（regular/bold/italic/bolditalic）。

    Release override: the four fonts shipped with the figure-reproduction
    package (``scripts/figures/fonts/``) take priority, so a clean clone
    reproduces the exact paper typography without TeX Live installed.
    """
    styles = ("regular", "bold", "italic", "bolditalic")
    found = []
    _bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    for st in styles:
        _p = os.path.join(_bundled, f"texgyreheros-{st}.otf")
        if os.path.exists(_p):
            found.append(_p)
    if len(found) == len(styles):
        return found
    found = []
    for exe in _kpsewhich_binaries():
        try:
            for st in styles:
                out = subprocess.run(
                    [exe, f"texgyreheros-{st}.otf"], capture_output=True,
                    text=True, timeout=15).stdout.strip()
                if out and os.path.exists(out):
                    found.append(out)
            if found:
                return found
        except Exception:
            continue
    # 没有 kpsewhich：直接在常见 texmf 树里找
    for root in (os.path.expanduser("~/texlive"), "/usr/share/texlive",
                 "/usr/local/texlive", "/opt/texlive"):
        for f in sorted(glob.glob(os.path.join(
                root, "**", "fonts", "opentype", "public", "tex-gyre",
                "texgyreheros-*.otf"), recursive=True)):
            found.append(f)
        if found:
            return found
    return found


def use_nature_fonts(verbose=True):
    """注册并启用 Helvetica 系字体 + 同源 mathtext，返回选中的字体名。"""
    global FAMILY
    families = _installed_families()
    if not any(f in families for f in _PREFERRED[:6]):
        # 系统里没有 Helvetica 系：注册 TeX Gyre Heros
        for f in _texgyre_heros_files():
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
        families = _installed_families()
    chosen = next((f for f in _PREFERRED if f in families), "DejaVu Sans")
    FAMILY = chosen

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen] + [f for f in _PREFERRED if f != chosen],
        # 数学符号与正文同源（默认 mathtext 走 DejaVu，和正文"两套字"）
        "mathtext.fontset": "custom",
        "mathtext.rm": chosen,
        "mathtext.it": f"{chosen}:italic",
        "mathtext.bf": f"{chosen}:bold",
        "mathtext.sf": chosen,
        "mathtext.cal": f"{chosen}:italic",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if verbose:
        print(f"[typography] font = {chosen}")
        if chosen == "DejaVu Sans":
            print("[typography] 警告：未找到 Helvetica 系字体，仍用 DejaVu")
    return chosen


def enforce_glyph_floor(fig, floor_pt=GLYPH_FLOOR_PT, verbose=True):
    """把含 mathtext 且父字号过小的 Text 提到 5 pt 底线以上。

    只处理带 ``$`` 的 Text（$H_s$、$10^{-1}$、$m\\ s^{-1}$ 等），纯文本的
    6 pt 刻度不受影响，所以不会打乱既有的字号层级。必须在一轮 ``draw()``
    之后调用，否则刻度标签的 Text 还没有被创建。

    参数
    ----
    fig : matplotlib.figure.Figure
    floor_pt : 目标底线（pt），默认 5
    verbose : 打印实际被提升的 Text 数量

    返回
    ----
    int : 被提升的 Text 数量
    """
    import matplotlib.text as mtext

    min_pt = max(MATH_MIN_PT, floor_pt / MATHTEXT_SHRINK)
    figs_done, raised = set(), 0
    for t in fig.findobj(mtext.Text):
        try:
            s = t.get_text()
            size = float(t.get_fontsize())
        except Exception:
            continue
        if not s or "$" not in s:
            continue
        if size < min_pt - 1e-6:
            t.set_fontsize(min_pt)
            figs_done.add(s)
            raised += 1
    if verbose and raised:
        print(f"[typography] glyph floor: raised {raised} mathtext items "
              f"< {min_pt:.2f} pt -> {min_pt:.1f} pt ({len(figs_done)} unique)")
    return raised
