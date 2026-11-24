"""
figures.py - presentation figures for Chapter 2.

  coloured_text_figure : text with every token shaded green / red
                         (human text vs watermarked text, side by side)
  probability_bars     : top next-token probabilities before vs after the boost

Works with any tokenizer: pass the token STRINGS (with their leading spaces)
and the green/red flags from watermark_core.green_flags.
Green tokens are also drawn in bold, so the figure still works for
colour-blind viewers and in black-and-white print.
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

GREEN_FILL = "#9fdcb0"
RED_FILL = "#f6c1b8"
NONE_FILL = "#e6e6e6"
INK = "#1f2933"
MUTED = "#6b7280"
BEFORE = "#c4c9d0"
AFTER = "#33658a"


def _draw_tokens(ax, tokens, flags, fontsize=11, y_top=0.86, max_width=0.98):
    """Lay out tokens like a paragraph. The figure layout must be FINAL before
    calling this, because positions are measured in axes coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    def extent(artist):
        (x0, y0), (x1, y1) = inv.transform(artist.get_window_extent(renderer))
        return x0, y0, x1, y1

    probe = ax.text(0, 0, "Mg", fontsize=fontsize, transform=ax.transAxes)
    _, y0, _, y1 = extent(probe)
    probe.remove()
    line_h = 1.75 * (y1 - y0)
    a = ax.text(0, 0, "a a", fontsize=fontsize, transform=ax.transAxes)
    b = ax.text(0, 0, "aa", fontsize=fontsize, transform=ax.transAxes)
    space_w = (extent(a)[2] - extent(a)[0]) - (extent(b)[2] - extent(b)[0])
    a.remove(); b.remove()
    x, y = 0.0, y_top
    for tok, flag in zip(tokens, flags):
        lead_space = tok[:1] in (" ", "\n")
        word = tok.strip() or "\u00b7"
        if lead_space and x > 0:
            x += space_w
        fill = NONE_FILL if flag is None else (GREEN_FILL if flag else RED_FILL)
        txt = ax.text(x, y, word, fontsize=fontsize, color=INK, weight="bold" if flag else "normal",
                      transform=ax.transAxes, va="baseline", ha="left", zorder=3)
        x0, b0, x1, b1 = extent(txt)
        if x1 > max_width and x > 0:            # wrap to the next line
            x, y = 0.0, y - line_h
            txt.set_position((x, y))
            x0, b0, x1, b1 = extent(txt)
        ax.add_patch(FancyBboxPatch((x0 - 0.001, b0 - 0.006), (x1 - x0) + 0.002, (b1 - b0) + 0.012,
                                    boxstyle="round,pad=0,rounding_size=0.003",
                                    facecolor=fill, edgecolor="none",
                                    transform=ax.transAxes, zorder=2))
        x = x1 + 0.0015
    return y, line_h


def coloured_text_figure(panels, path, n_show=70, fontsize=11):
    """panels: list of dicts with keys title, tokens (strings), flags, stats (dict from detect).
    Shows the first n_show tokens of each text; the stats line covers the whole text."""
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 3.5))
    if len(panels) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03, wspace=0.08)
    fig.canvas.draw()                              # fix the layout BEFORE measuring text
    for ax, p in zip(axes, panels):
        ax.set_axis_off()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.0, 0.99, p["title"], fontsize=fontsize + 3, weight="bold", color=INK,
                transform=ax.transAxes, va="top")
        y, line_h = _draw_tokens(ax, p["tokens"][:n_show], p["flags"][:n_show], fontsize=fontsize, y_top=0.80)
        shown = [f for f in p["flags"][:n_show] if f is not None]
        s = p["stats"]
        ax.text(0.0, y - 1.6 * line_h,
                f"Shown: first {len(p['tokens'][:n_show])} tokens ({sum(shown)} green).   "
                f"Whole text: {s['G']} of {s['T']} green ({100 * s['green_frac']:.0f}%), "
                f"z = {s['z']:.2f}, exact p = {s['p_exact']:.2g}",
                fontsize=fontsize - 1, color=MUTED, transform=ax.transAxes)
    handles = [Patch(facecolor=GREEN_FILL, label="green token (bold)"),
               Patch(facecolor=RED_FILL, label="red token")]
    axes[-1].legend(handles=handles, loc="upper right", ncol=2, frameon=False,
                    bbox_to_anchor=(1.0, 1.03), fontsize=fontsize - 1)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def probability_bars(rows, green_before, green_after, context, delta, path):
    """rows: list of dicts token, green, p_before, p_after (top tokens by p_before)."""
    labels = [f"{r['token'].strip() or repr(r['token'])}\n({'G' if r['green'] else 'R'})" for r in rows]
    xs = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([x - w / 2 - 0.01 for x in xs], [100 * r["p_before"] for r in rows], w,
           color=BEFORE, label="before watermark")
    ax.bar([x + w / 2 + 0.01 for x in xs], [100 * r["p_after"] for r in rows], w,
           color=AFTER, label=f"after adding δ = {delta:g} to green tokens")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=11)
    for tick, r in zip(ax.get_xticklabels(), rows):
        tick.set_bbox(dict(facecolor=GREEN_FILL if r["green"] else RED_FILL,
                           edgecolor="none", boxstyle="round,pad=0.25"))
    ax.set_ylabel("Probability of being the next token (%)", color=INK)
    ctx = textwrap.shorten(context, 70, placeholder="…")
    ax.set_title(f"Next token after “…{ctx[-60:]}”\n"
                 f"Total chance of a green token: {100 * green_before:.0f}% → {100 * green_after:.0f}%",
                 fontsize=13, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color="#eeeeee"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11)
    fig.text(0.01, 0.005, "(G) = green token, (R) = red token for this step", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
