# -*- coding: utf-8 -*-
"""
Shared RF plotting helpers.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


def plot_rf_grid(
    rf_per_array,
    array2area,
    title,
    vlim=(-0.5, 0.5),
    params_per_array=None,
    show_center_text=True,
    save_dir=None,
    basename="RFs",
    dpi=300,
    cmap="bwr",
):
    """
    Plot RFs per array with optional center markers.

    rf_per_array: np.ndarray, shape (N_arrays, H, W)
    array2area: dict mapping 1-based array id -> area name
    title: figure title
    params_per_array: dict with keys RFX/RFY (optional)
    """
    n_arrays, H, W = rf_per_array.shape
    n_cols = min(4, n_arrays)
    n_rows = math.ceil(n_arrays / n_cols)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows))
    fig.suptitle(title, fontsize=18)

    gs = GridSpec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.4,
        hspace=0.5,
    )

    if vlim is None:
        vmax = np.max(np.abs(rf_per_array))
        vmin = -vmax
    else:
        vmin, vmax = vlim

    for idx in range(n_arrays):
        r = idx // n_cols
        c = idx % n_cols
        ax = fig.add_subplot(gs[r, c])

        rf = rf_per_array[idx]
        im = ax.imshow(rf, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

        rect = Rectangle(
            xy=(-0.5, -0.5),
            width=W,
            height=H,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        if params_per_array is not None:
            cx = params_per_array["RFX"][idx]
            cy = params_per_array["RFY"][idx]
            if np.isfinite([cx, cy]).all():
                ax.plot(cx, cy, marker="o", markersize=4, color="black")
                if show_center_text:
                    ax.text(
                        0.02, 0.02, f"({cx:.1f},{cy:.1f})",
                        transform=ax.transAxes, fontsize=8,
                        color="black", ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"),
                    )

        array_num = idx + 1
        area = array2area.get(array_num, "unknown")
        ax.set_title(f"Array {array_num} - {area}", fontsize=11, pad=10)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("corr(pixel, MUA)")

    plt.tight_layout(rect=[0, 0, 0.96, 0.95])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"{basename}.png")
        pdf_path = os.path.join(save_dir, f"{basename}.pdf")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_rf_grid] Saved PNG to: {png_path}")
        print(f"[plot_rf_grid] Saved PDF to: {pdf_path}")

    plt.show()
