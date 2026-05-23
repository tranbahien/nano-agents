"""Figure: the ReAct agent loop as an extended MDP.

A schematic showing the think -> act -> observe cycle, with the action
space augmented to include tool calls. Annotates the correspondence to
the POMDP framework from Post 1d.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def box(ax, xy, w, h, text, color, fontsize=11, fontweight="normal",
         textcolor="#222"):
    rect = patches.FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor="#222",
        linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center", color=textcolor,
            fontsize=fontsize, fontweight=fontweight, zorder=4)


def arrow(ax, src, dst, label="", color="#222", curve=0.0, fontsize=9,
           loff=(0, 0)):
    ax.annotate("", xy=dst, xytext=src,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                  connectionstyle=f"arc3,rad={curve}",
                                  shrinkA=12, shrinkB=12))
    if label:
        mx, my = 0.5 * (src[0] + dst[0]) + loff[0], 0.5 * (src[1] + dst[1]) + loff[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    # Agent (LM policy) on the left.
    box(ax, (2.2, 3.5), 2.6, 1.6,
        "Agent  $\\pi_\\theta$\n(language model)", "#e8f0fe",
        fontsize=12, fontweight="bold")

    # Action types stacked in the middle.
    box(ax, (5.0, 5.4), 2.2, 0.9, "THINK\n(reasoning token)", "#fff2cc",
        fontsize=10)
    box(ax, (5.0, 3.5), 2.2, 0.9, "ACT\n(tool call)", "#fde2e2", fontsize=10)
    box(ax, (5.0, 1.6), 2.2, 0.9, "ANSWER\n(final output)", "#e0f0e0",
        fontsize=10)

    # Environment / tools on the right.
    box(ax, (8.2, 3.5), 2.4, 2.4,
        "Tools / Environment\n\ncalculator\nsearch\ncode exec\nlookup",
        "#f0f0f0", fontsize=10)

    # Arrows: agent -> action types.
    arrow(ax, (3.5, 3.9), (3.9, 5.2), color="#bbab5f")
    arrow(ax, (3.5, 3.5), (3.9, 3.5), color="#c44e52")
    arrow(ax, (3.5, 3.1), (3.9, 1.8), color="#55a467")

    # ACT -> tools, tools -> agent (observation).
    arrow(ax, (6.1, 3.5), (7.0, 3.5), label="invoke", color="#c44e52",
           fontsize=9, loff=(0, 0.25))
    arrow(ax, (7.0, 2.6), (3.5, 2.6), label="observation (tool result)",
           color="#3a7ebf", curve=0.3, fontsize=10, loff=(0, -0.5))

    # THINK loops back to agent.
    arrow(ax, (3.9, 5.6), (3.4, 4.1), color="#bbab5f", curve=0.3)

    # ANSWER exits.
    arrow(ax, (6.1, 1.6), (8.0, 1.6), label="reward", color="#55a467",
           fontsize=10, loff=(0, 0.25))
    ax.text(8.6, 1.6, "$r$", fontsize=14, color="#55a467", va="center")

    # POMDP correspondence annotation.
    ax.text(5.0, 0.4,
            "Extended MDP: state = (prompt, history of thoughts + observations);  "
            "action $\\in$ {emit token} $\\cup$ {tool calls};  "
            "tool result = observation.",
            ha="center", fontsize=10, style="italic", color="#444")

    ax.set_title(
        "Tool use turns the LM into an agent. The ReAct loop: think, act "
        "(call a tool), observe the result, repeat, then answer.",
        fontsize=12)
    fig.tight_layout()
    out = Path("figures/03b_react_loop.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
