# -*- coding: utf-8 -*-

"""
* Why upside-down in RF plots?

Bare coordinate-system plot for RF arena (no RF points/ellipses).
"""
# %% Config
PIX_PER_DEG = 25.8601
monkey = "monkeyN"
shuffle_mapping = False
if shuffle_mapping:
    shuffle_suffix = 'shuffled_order'
else:
    shuffle_suffix = 'true_order'

# %%
wd = r"E:\radboud\Masters Thesis"
# image-side tree
image_side_dir = join(wd, 'data', 'source data', 'image data')
things_dir = join(image_side_dir, 'THINGS')
object_images_dir = os.path.join(things_dir,"images_THINGS", "object_images")
# ephys side tree
ephys_side_dir = join(wd, 'data', 'source data', 'neural data')
tvsd_dir = join(ephys_side_dir, 'TVSD')
log_path = join(tvsd_dir, monkey, '_logs')
image_MUA_mapping = join(log_path, 'things_imgs.mat') # mapping
normMUA_path = join(tvsd_dir, monkey, 'THINGS_normMUA.mat')
# derivatives tree
derivatives_ephys_dir = join(wd, 'derivatives', 'neural data', 'TVSD')
derivatives_rf_dir = join(derivatives_ephys_dir, monkey, 'ReceptiveFields')
# ana tree
ana_dir = join(wd, 'analysis', 'TVSD')
ana_monkey_dir = join(ana_dir, monkey)
save_dir =  join(ana_monkey_dir, 'Exploration', 'ReceptiveFields', 'STA')
# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from os.path import join
import os

def load_rf_centers_csv(csv_path):
    """
    Load RF centers saved as CSV with header: array,rfx,rfy.
    Returns array_ids, rfx, rfy.
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    centers_dict = {
        "array": data[:, 0].astype(int),
        "rfx": (((data[:, 1]) * 7.8) - 100) / PIX_PER_DEG, # / PIX_PER_DEG,
        "rfy": (((data[:, 2]) * 7.8) - 100) / PIX_PER_DEG * -1 # / PIX_PER_DEG,  # invert y-axis
    }

    return centers_dict

#%%
def plot_rf_arena(
    xlim=(-2, 8),
    ylim=(-7.5, 3.1),
    center=(0, 0),
    radius=8,
    num_circles=4,
    figsize=(6, 6),
    dpi=200,
    title="RF Arena",
    centers=None,
    centers_color="black",
    centers_size=20,
    centers_cmap="rainbow",
):
    """
    Plot the RF coordinate system (axes + concentric circles),
    optionally overlaying RF centers from a CSV.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Concentric circles
    for idx in range(num_circles):
        circle = patches.Circle(
            center,
            radius * (idx + 1) / num_circles,
            edgecolor="gray",
            linewidth=0.5,
            fill=False,
            linestyle="dashed",
        )
        ax.add_patch(circle)

    # Axes lines
    ax.axvline(0, color="gray", linestyle="dashed", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="dashed", linewidth=0.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if centers is not None:
        rfx = centers["rfx"]
        rfy = centers["rfy"]
        array_ids = centers["array"]
        cmap = plt.get_cmap(centers_cmap)
        order = (array_ids - 1).astype(float)
        norm = (order - np.nanmin(order)) / (np.nanmax(order) - np.nanmin(order) + 1e-9)
        colors = cmap(norm)
        ax.scatter(rfx, rfy, s=centers_size, color=colors, marker="x")

        array_ids_unique = np.unique(array_ids)
        array_color_map = {
            int(a): tuple(colors[np.where(array_ids == a)[0][0]])
            for a in array_ids_unique
        }
        legend_handles = [
           plt.Line2D([0], [0], marker="x", color=array_color_map[a],
                       markerfacecolor="none",
                       markeredgecolor=array_color_map[a], markersize=6,
                       label=f"{a}")
           
            for a in array_ids_unique
        ]
        ax.legend(handles=legend_handles, title="Array", loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                  fontsize=8, title_fontsize=9, frameon=False)

        print("[RF_ARENA] centers_loaded:", len(centers["array"]))
        print("[RF_ARENA] rfx min/med/max:",
              np.nanmin(rfx), np.nanmedian(rfx), np.nanmax(rfx))
        print("[RF_ARENA] rfy min/med/max:",
              np.nanmin(rfy), np.nanmedian(rfy), np.nanmax(rfy))
        print("[RF_ARENA] centers_cmap:", centers_cmap)

    # Diagnostics
    print("[RF_ARENA] center:", center)
    print("[RF_ARENA] radius:", radius)
    print("[RF_ARENA] num_circles:", num_circles)
    print("[RF_ARENA] xlim:", xlim)
    print("[RF_ARENA] ylim:", ylim)

    plt.tight_layout()
    plt.show()


def plot_rf_arena_compare(
    centers_by_method,
    xlim=(-2, 8),
    ylim=(-7.5, 3.1),
    center=(0, 0),
    radius=8,
    num_circles=4,
    figsize=(6, 6),
    dpi=200,
    title="RF Arena - GLM vs STA",
):
    """
    Plot RF coordinate system and compare multiple center-estimation methods.

    centers_by_method: dict of {name: centers_dict}
      centers_dict should have keys: array, rfx, rfy.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Concentric circles
    for idx in range(num_circles):
        circle = patches.Circle(
            center,
            radius * (idx + 1) / num_circles,
            edgecolor="gray",
            linewidth=0.5,
            fill=False,
            linestyle="dashed",
        )
        ax.add_patch(circle)

    ax.axvline(0, color="gray", linestyle="dashed", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="dashed", linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    markers = ["o", "x", "^", "s", "D", "v", "*", "P"]
    cmap = plt.get_cmap("rainbow")

    all_arrays = sorted({int(a) for centers in centers_by_method.values()
                         for a in centers["array"]})
    order = np.array(all_arrays, dtype=float) - 1
    norm = (order - np.nanmin(order)) / (np.nanmax(order) - np.nanmin(order) + 1e-9)
    array_color_map = {a: tuple(cmap(norm[i])) for i, a in enumerate(all_arrays)}

    method_names = list(centers_by_method.keys())
    method_centers = {}

    for idx, (name, centers) in enumerate(centers_by_method.items()):
        rfx = centers["rfx"]
        rfy = centers["rfy"]
        array_ids = centers["array"]
        marker = markers[idx % len(markers)]
        colors = np.array([array_color_map[int(a)] for a in array_ids])
        valid = np.isfinite(rfx) & np.isfinite(rfy)

        ax.scatter(rfx[valid], rfy[valid], s=20, color=colors[valid], marker=marker)

        missing_arrays = sorted(set(array_ids[~valid].astype(int)))
        if missing_arrays:
            print(f"[RF_ARENA] {name} missing arrays (NaN centers):", missing_arrays)

        print(f"[RF_ARENA] {name} centers_loaded:", len(array_ids))
        print(f"[RF_ARENA] {name} centers_plotted:", int(np.sum(valid)))
        print(f"[RF_ARENA] {name} rfx min/med/max:",
              np.nanmin(rfx[valid]), np.nanmedian(rfx[valid]), np.nanmax(rfx[valid]))
        print(f"[RF_ARENA] {name} rfy min/med/max:",
              np.nanmin(rfy[valid]), np.nanmedian(rfy[valid]), np.nanmax(rfy[valid]))

        method_centers[name] = {
            int(a): (float(x), float(y))
            for a, x, y, ok in zip(array_ids, rfx, rfy, valid)
            if ok
        }

    if len(method_names) >= 2:
        base = method_names[0]
        for other in method_names[1:]:
            shared = sorted(set(method_centers[base].keys()) & set(method_centers[other].keys()))
            for a in shared:
                x0, y0 = method_centers[base][a]
                x1, y1 = method_centers[other][a]
                ax.plot([x0, x1], [y0, y1],
                        color=array_color_map[a], linestyle=":", linewidth=0.8, alpha=0.35)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=array_color_map[a],
                   markeredgecolor=array_color_map[a],
                   markersize=6, label=f"{a}")
        for a in all_arrays
    ]
    array_legend = ax.legend(handles=legend_handles, title="Array", loc="upper left",
                             bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                             fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(array_legend)

    method_handles = [
        plt.Line2D([0], [0], marker=markers[idx % len(markers)],
                   color="black", linestyle="none",
                   markersize=6, label=name)
        for idx, name in enumerate(method_names)
    ]
    ax.legend(handles=method_handles, title="Method", loc="upper left",
              bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0,
              fontsize=8, title_fontsize=9, frameon=False)

    print("[RF_ARENA] center:", center)
    print("[RF_ARENA] radius:", radius)
    print("[RF_ARENA] num_circles:", num_circles)
    print("[RF_ARENA] xlim:", xlim)
    print("[RF_ARENA] ylim:", ylim)

    plt.tight_layout()
    plt.show()

# %% 
rf_lin_dir =  join(ana_monkey_dir, 'Exploration', 'ReceptiveFields', 'linear')

basename_sta = f"{monkey}_STA_RFs_params_{shuffle_suffix}"
basename_glm = f"{monkey}_GLM_RFs_params_{shuffle_suffix}"

sta_path = os.path.join(rf_lin_dir, "STA", f"{basename_sta}_rf_centers_per_array.csv")
glm_path = os.path.join(rf_lin_dir, "GLM", f"{basename_glm}_rf_centers_per_array.csv")

sta_centers = load_rf_centers_csv(sta_path)
glm_centers = load_rf_centers_csv(glm_path)

arena_centers = {
    "STA": sta_centers,
    "GLM": glm_centers
}
plot_rf_arena_compare(centers_by_method=arena_centers)

# %%
