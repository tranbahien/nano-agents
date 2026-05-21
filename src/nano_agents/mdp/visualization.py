"""Plotting helpers for gridworld value functions and policies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .environments import ACTIONS, GridWorld


def value_to_grid(V: np.ndarray, env: GridWorld, fill: float = np.nan) -> np.ndarray:
    """Lay the flat value vector V back onto the grid as a (rows, cols) array.
    Wall cells are filled with `fill` (default NaN, which matplotlib treats specially).
    """
    grid = np.full((env.rows, env.cols), fill)
    for s, i in env.state_to_idx.items():
        grid[s] = V[i]
    return grid


def plot_value(env: GridWorld, V: np.ndarray, ax=None, cmap="viridis",
                vmin=None, vmax=None, show_text: bool = True, fontsize: int = 9):
    """Plot a state-value function as a heatmap with optional text overlay."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    g = value_to_grid(V, env)
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    # Draw walls as dark cells.
    for wr, wc in env.walls:
        ax.add_patch(plt.Rectangle((wc - 0.5, wr - 0.5), 1, 1,
                                     facecolor="#222", zorder=2))
    # Draw terminals with a different border.
    for (tr, tc), tv in env.terminals.items():
        rect = plt.Rectangle((tc - 0.5, tr - 0.5), 1, 1,
                              facecolor="none", edgecolor="#c44e52",
                              linewidth=2.5, zorder=3)
        ax.add_patch(rect)
    if show_text:
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.walls:
                    continue
                # For terminal states, display the entry reward, not the (zero) terminal value.
                if (r, c) in env.terminals:
                    text = f"+{int(env.terminals[(r, c)])}"
                    txt_color = "#c44e52"
                else:
                    val = g[r, c]
                    text = f"{val:.2f}"
                    txt_color = "white" if val < (np.nanmin(g) + np.nanmax(g)) / 2 else "black"
                ax.text(c, r, text, ha="center", va="center",
                        color=txt_color, fontsize=fontsize, zorder=4,
                        fontweight="bold" if (r, c) in env.terminals else "normal")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_policy(env: GridWorld, pi: np.ndarray, V: np.ndarray | None = None,
                 ax=None, cmap: str = "viridis"):
    """Plot a deterministic policy as arrows, optionally over a value heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    if V is not None:
        g = value_to_grid(V, env)
        ax.imshow(g, cmap=cmap, interpolation="nearest", alpha=0.6)
    else:
        # Empty white background.
        ax.imshow(np.full((env.rows, env.cols), 0.0), cmap="Greys",
                   vmin=0, vmax=1, interpolation="nearest")
    for wr, wc in env.walls:
        ax.add_patch(plt.Rectangle((wc - 0.5, wr - 0.5), 1, 1,
                                     facecolor="#222", zorder=2))
    for (tr, tc), tv in env.terminals.items():
        ax.add_patch(plt.Rectangle((tc - 0.5, tr - 0.5), 1, 1,
                                     facecolor="none", edgecolor="#c44e52",
                                     linewidth=2.5, zorder=3))
        ax.text(tc, tr, f"{tv:+.0f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#c44e52", zorder=4)
    arrow_len = 0.32
    for s, i in env.state_to_idx.items():
        if env.is_terminal(s):
            continue
        r, c = s
        a = int(pi[i])
        dr, dc = ACTIONS[a]
        ax.annotate("", xy=(c + arrow_len * dc, r + arrow_len * dr),
                     xytext=(c - arrow_len * dc, r - arrow_len * dr),
                     arrowprops=dict(arrowstyle="->", color="white",
                                      lw=2.0), zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
