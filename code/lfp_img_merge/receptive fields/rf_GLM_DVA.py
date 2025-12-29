#%%
# import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Plot function (verbatim-ish)
# -----------------------------
def plot_all_RFs(horizontal_pos_MaxResponses_mean, vertical_pos_MaxResponses_mean,
                 sizes, goodIDs, allColors, markerSize, offSet, text=False):
    AREA = ['V1'] * (64 * 8) + ['V4'] * (64 * 4) + ['IT'] * (64 * 4)

    stdCoeff = 1
    xOffset, yOffset = offSet

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=200)

    for area, ax in zip(['V1', 'V4', 'IT'], axs.flatten()):
        ax.set_title(area, size=20)

        # Ellipses
        for i in range(1024):
            if goodIDs[i] and AREA[i] == area:
                if area == 'V1':
                    hatch, alpha = None, 0.3
                elif area == 'V4':
                    hatch, alpha = '//////', 0.03
                else:
                    hatch, alpha = '///', 0.01

                color = tuple(allColors[i])
                ellipse = patches.Ellipse(
                    (horizontal_pos_MaxResponses_mean[i] + xOffset,
                     vertical_pos_MaxResponses_mean[i] + yOffset),
                    width=sizes[i] * stdCoeff,
                    height=sizes[i] * stdCoeff,
                    edgecolor=color, linewidth=0.5,
                    hatch=hatch,
                    facecolor=color,
                    alpha=alpha
                )
                ax.add_patch(ellipse)

        # Centers
        for i in range(1024):
            if AREA[i] == area:
                if area == 'V1':
                    marker, s = 'o', 5 * markerSize
                elif area == 'V4':
                    marker, s = 'v', 10 * markerSize
                else:
                    marker, s = 'x', 10 * markerSize

                if goodIDs[i]:
                    color = tuple(allColors[i])
                    ax.scatter(horizontal_pos_MaxResponses_mean[i] + xOffset,
                               vertical_pos_MaxResponses_mean[i] + yOffset,
                               color=color, marker=marker,
                               edgecolors='black', linewidth=0.5, s=s)

        # Concentric circles + axes
        center = (0, 0)
        radius = 8
        num_circles = 4
        for k in range(num_circles):
            circle = patches.Circle(center, radius * (k + 1) / num_circles,
                                    edgecolor='gray', linewidth=0.5,
                                    fill=False, linestyle='dashed')
            ax.add_patch(circle)

        ax.axvline(0, color='gray', linestyle='dashed', linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='dashed', linewidth=0.5)
        ax.set_xlim(-2, 8)
        ax.set_ylim(-7.5, 3.1)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # All areas subplot
    ax = axs[1, 1]
    ax.set_title('All Areas')

    center = (0, 0)
    radius = 8
    num_circles = 4

    for i in range(1024):
        if goodIDs[i]:
            area = AREA[i]
            if area == 'V1':
                hatch, alpha = None, 0.3
                marker, s = 'o', 5 * markerSize
            elif area == 'V4':
                hatch, alpha = '//////', 0.03
                marker, s = 'v', 10 * markerSize
            else:
                hatch, alpha = '///', 0.01
                marker, s = 'x', 10 * markerSize

            color = tuple(allColors[i])
            ellipse = patches.Ellipse(
                (horizontal_pos_MaxResponses_mean[i] + xOffset,
                 vertical_pos_MaxResponses_mean[i] + yOffset),
                width=sizes[i] * stdCoeff,
                height=sizes[i] * stdCoeff,
                edgecolor=color, linewidth=0.5,
                hatch=hatch,
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(ellipse)

            ax.scatter(horizontal_pos_MaxResponses_mean[i] + xOffset,
                       vertical_pos_MaxResponses_mean[i] + yOffset,
                       color=color, marker=marker,
                       edgecolors='black', linewidth=0.5, s=s)

    for k in range(num_circles):
        circle = patches.Circle(center, radius * (k + 1) / num_circles,
                                edgecolor='gray', linewidth=0.5,
                                fill=False, linestyle='dashed')
        ax.add_patch(circle)

    ax.axvline(0, color='gray', linestyle='dashed', linewidth=0.5)
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=0.5)
    ax.set_xlim(-2, 8)
    ax.set_ylim(-7.5, 3.1)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# -----------------------------
# CONFIG: paths
# -----------------------------
GLM_RF_PATH = r"E:\radboud\Masters Thesis\derivatives\neural data\TVSD\monkeyN\ReceptiveFields\linear\monkeyN_GLM_RFs_1000imgs_ordered_alpha0.1_l1r0.8.npz"
MAPPING_PKL = r"C:\Users\Radovan\OneDrive\Radboud\a_Internship\Antonio Lonzano\root\SlavsForSight\code\NIN_canon\MAPPING\results\mapping_MrNilson.pkl"

assert os.path.exists(GLM_RF_PATH), f"Missing GLM file: {GLM_RF_PATH}"
assert os.path.exists(MAPPING_PKL), f"Missing mapping pkl: {MAPPING_PKL}"

# -----------------------------
# LOAD mapping.pkl (for colors)
# -----------------------------
with open(MAPPING_PKL, "rb") as f:
    map_nilson = pickle.load(f)

allColors = map_nilson["arrayColor"]
assert allColors.shape == (1024, 3), f"Expected arrayColor (1024,3), got {allColors.shape}"

# -----------------------------
# LOAD GLM results
# -----------------------------
glm = np.load(GLM_RF_PATH, allow_pickle=True)

H = int(glm["H"])
W = int(glm["W"])
PIX_PER_DEG = float(glm["PIX_PER_DEG_THINGS"])
E = int(glm["E"])

# These exist if you saved them
all_centrex = glm["all_centrex"]  # (E,)
all_centrey = glm["all_centrey"]  # (E,)
all_szx     = glm["all_szx"]      # (E,)
all_szy     = glm["all_szy"]      # (E,)

print("Loaded GLM:")
print("  E,H,W:", E, H, W)
print("  PIX_PER_DEG:", PIX_PER_DEG)
print("  centrex/centrey shapes:", all_centrex.shape, all_centrey.shape)

# -----------------------------
# Build 1024-length arrays expected by plot_all_RFs
# -----------------------------
x_deg_full = np.full(1024, np.nan, dtype=np.float32)
y_deg_full = np.full(1024, np.nan, dtype=np.float32)
sizes_full = np.full(1024, np.nan, dtype=np.float32)

# recenter + flip y (pixel -> DVA)
cx = (W - 1) / 2.0
cy = (H - 1) / 2.0

x_deg = (all_centrex - cx) / PIX_PER_DEG
y_deg = -(all_centrey - cy) / PIX_PER_DEG

sx_deg = all_szx / PIX_PER_DEG
sy_deg = all_szy / PIX_PER_DEG

# single "size" scalar for ellipse width/height
size_deg = 0.5 * (sx_deg + sy_deg)

# place into 1024 vector (assumes GLM channel order == mapping order)
# if E==1024 this is a no-op; if E<1024 we fill first E
x_deg_full[:E] = x_deg
y_deg_full[:E] = y_deg
sizes_full[:E] = size_deg

goodIDs = np.isfinite(x_deg_full) & np.isfinite(y_deg_full) & np.isfinite(sizes_full)

print("\nDescriptives (finite only):")
print("  x_deg:  mean/std/min/max:",
      float(np.nanmean(x_deg_full)), float(np.nanstd(x_deg_full)),
      float(np.nanmin(x_deg_full)), float(np.nanmax(x_deg_full)))
print("  y_deg:  mean/std/min/max:",
      float(np.nanmean(y_deg_full)), float(np.nanstd(y_deg_full)),
      float(np.nanmin(y_deg_full)), float(np.nanmax(y_deg_full)))
print("  size:   mean/std/min/max:",
      float(np.nanmean(sizes_full)), float(np.nanstd(sizes_full)),
      float(np.nanmin(sizes_full)), float(np.nanmax(sizes_full)))
print("  goodIDs count:", int(goodIDs.sum()), "/ 1024")

# -----------------------------
# Plot in DVA using their assumptions
# -----------------------------
plot_all_RFs(
    horizontal_pos_MaxResponses_mean=x_deg_full,
    vertical_pos_MaxResponses_mean=y_deg_full,
    sizes=sizes_full,
    goodIDs=goodIDs,
    allColors=allColors,
    markerSize=8,
    offSet=[0, 0],
    text=False
)

