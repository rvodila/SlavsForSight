# %% ---------- CONFIG ----------
PIX_PER_DEG_THINGS = 25.8601  # pixels per degree, from mapping.py
monkey = 'monkeyN' # monkeyF
out_size = (64, 64)  # RF grid resolution: 64x64 "sensors" (stimulus pixels); 128 is already crashing my system
n_subset = 10000 # random subset of >22248 images in train set
E_SUBSET = 512 # None
if E_SUBSET is not None:
    assert E_SUBSET % 64 == 0
shuffle_mapping = False # Shuffle true mapping of MUA - image. acts as sanity check.
reg = None #l1, l2, elastic
VERBOSE = False
# ElasticNet for a balance between sparsity and smoothness
# l1 ~
alpha = 0.2
l1_ratio = 0.8
# Vis
vlim = (-1e-3, 1e-3)# 
# %%
# ---------- IMPORTS ----------
import os
import numpy as np
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from os.path import join
# ---------- DIR ----------
wd = r"E:\radboud\Masters Thesis" # r"C:\Users\Radovan\OneDrive\Radboud\a_Internship\Antonio Lonzano\root\SlavsForSight"
# image-side tree
image_side_dir = join(wd, 'data','source data', 'image data')
things_dir = join(image_side_dir, 'THINGS')
object_images_dir = os.path.join(things_dir,"images_THINGS", "object_images")
# ephys side tree
ephys_side_dir = join(wd, 'data','source data', 'neural data')
tvsd_dir = join(ephys_side_dir, 'TVSD')
log_path = join(tvsd_dir, monkey, '_logs')
image_MUA_mapping = join(log_path, 'things_imgs.mat') # mapping
normMUA_path = join(tvsd_dir, monkey, 'THINGS_normMUA.mat')
# derivatives tree
derivatives_ephys_dir = join(wd, 'data''derivatives', 'neural data', 'TVSD')
derivatives_rf_dir = join(derivatives_ephys_dir, monkey, 'ReceptiveFields')
# ana tree
ana_dir = join(wd, 'analysis', 'TVSD')
ana_monkey_dir = join(ana_dir, monkey)
# %%
# ---------- image side helpers ----------
def prepare_THINGS_objects(image_MUA_mapping, verbose = False):
    with h5py.File(image_MUA_mapping, "r") as f:
        train_group = f["train_imgs"]
        test_group  = f["test_imgs"]

        class_refs       = train_group["class"][()]        # (22248,)
        local_path_refs  = train_group["local_path"][()]   # (22248,)
        things_path_refs = train_group["things_path"][()]  # (22248,)

        print("class_refs shape:", class_refs.shape, "dtype:", class_refs.dtype)
        print("local_path_refs shape:", local_path_refs.shape, "dtype:", local_path_refs.dtype)
        print("things_path_refs shape:", things_path_refs.shape, "dtype:", things_path_refs.dtype)

        def unwrap_ref(ref):
            """
            Unwrap a reference that may be:
            - an h5py.Reference
            - a 0-d or 1-d numpy array containing a Reference
            and return a true h5py.Reference.
            """
            # Direct reference
            if isinstance(ref, h5py.Reference):
                return ref

            # 0-d / 1-d numpy array wrapping the reference
            arr = np.array(ref)
            if arr.dtype == object and arr.size > 0:
                inner = arr.ravel()[0]
                if isinstance(inner, h5py.Reference):
                    return inner
                # if it's still an array, recurse once
                if isinstance(inner, np.ndarray):
                    return unwrap_ref(inner)

            # Fallback: just try to treat ref as-is (will error if wrong)
            return ref

        def dset_to_str(dset):
            """
            Convert a dataset (often MATLAB char array) to a Python string.
            Handles:
            - bytes
            - numpy arrays of 'S'/'U'
            - numpy arrays of uint16/int16/int32 char codes
            """
            data = dset[()]  # grab contents

            # plain python types
            if isinstance(data, bytes):
                return data.decode("utf-8")
            if isinstance(data, str):
                return data

            arr = np.array(data)

            # string-like arrays
            if arr.dtype.kind in {"S", "U"}:
                flat = arr.ravel()
                parts = []
                for x in flat:
                    if isinstance(x, (bytes, bytearray)):
                        parts.append(x.decode("utf-8"))
                    else:
                        parts.append(str(x))
                return "".join(parts)

            # numeric char codes (MATLAB-style)
            if arr.dtype in (np.uint16, np.int16, np.int32, np.int64):
                chars = [chr(int(c)) for c in arr.ravel() if int(c) != 0]
                return "".join(chars)

            # last resort
            return str(arr)

        def refs_to_str_array(refs):
            strings = []
            for ref in refs:
                href = unwrap_ref(ref)
                dset = f[href]           # <--- this is where we previously passed an ndarray
                s = dset_to_str(dset)
                strings.append(s)
            return np.array(strings, dtype=object)

        train_class       = refs_to_str_array(class_refs)
        train_local_path  = refs_to_str_array(local_path_refs)
        train_things_path = refs_to_str_array(things_path_refs)
    if verbose:
        print('structure overview for index 0:')
        print("train_class[0]:      ", train_class[0])
        print("train_local_path[0]: ", train_local_path[0])
        print("train_things_path[0]:", train_things_path[0])
        print("n_train:", len(train_things_path))

    return train_class, train_local_path, train_things_path 

from PIL import Image
def load_image_downsample_gray(rel_path, root, size, verbose=False):
    """
    Load an image given a relative path, convert to grayscale, resize, and
    normalize to [0, 1].

    Args:
        rel_path: relative path to image (e.g. "aardvark/aardvark_01b.jpg")
        root: root directory where images live
        size: (H, W) tuple for resizing
        verbose: if True, print debug info

    Returns:
        arr: np.ndarray of shape (H, W), float32 in [0, 1]
    """
    img_path = os.path.join(root, rel_path)

    if verbose:
        print(f"[load_image] Loading: {img_path}")

    img = Image.open(img_path).convert("L")  # grayscale
    orig_w, orig_h = img.size
    #print(img.size)
    if verbose:
        print(f"[load_image] Original size: {orig_w}x{orig_h} -> resizing to {size[0]}x{size[1]}")

    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr /= 255.0  # scale to [0,1]

    if verbose:
        print(f"[load_image] After resize: shape={arr.shape}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}")

    return arr

def get_image_size(rel_path, root):
    """
    Return original (W, H) image size without resizing.
    """
    img_path = os.path.join(root, rel_path)
    img = Image.open(img_path)
    return img.size
# %%
# ---------- RF HELPERS ----------
def build_stimulus_and_response(
    image_paths,
    responses,
    root,
    size,
    idxs=None,
    verbose=True,
):
    """
    Build stimulus matrix X and response matrix R.

    Args:
        image_paths: array-like of length N_all with relative image paths.
        responses: np.ndarray of shape (N_all, E) with MUA responses.
        root: root directory for images.
        size: (H, W) for downsampled grayscale images.
        idxs: optional indices of images to use; if None, use all.
        verbose: if True, print debug info.

    Returns:
        X: np.ndarray of shape (N, D)  (N images, D pixels)
        R: np.ndarray of shape (N, E)  (N images, E electrodes)
        img_shape: tuple (H, W)
    """
    image_paths = np.asarray(image_paths)
    N_all = len(image_paths)

    if idxs is None:
        idxs = np.arange(N_all)
        if verbose:
            print(f"[build_SR] Using ALL {N_all} training images.")
    else:
        idxs = np.asarray(idxs)
        if verbose:
            print(f"[build_SR] Using subset of {len(idxs)} images out of {N_all} total.")

    N = len(idxs)
    E = responses.shape[1]

    # Load first image once to infer shape
    if verbose:
        print(f"[build_SR] Loading first example image at index {idxs[0]} for shape inference.")
    example_img = load_image_downsample_gray(image_paths[idxs[0]], root=root, size=size, verbose=verbose)
    H, W = example_img.shape
    D = H * W

    if verbose:
        print(f"[build_SR] Image shape: {H}x{W} -> D={D} pixels per image.")
        print(f"[build_SR] Number of electrodes E={E}")
        print(f"[build_SR] Allocating X with shape (N={N}, D={D}), R with shape (N={N}, E={E})")

    X = np.zeros((N, D), dtype=np.float32)
    R = np.zeros((N, E), dtype=np.float32)

    # Fill first row with example_img so we don't reload it
    X[0, :] = example_img.flatten()
    R[0, :] = responses[idxs[0], :]

    # Loop over remaining indices
    for k, i in enumerate(idxs[1:], start=1):
        if verbose and (k % max(1, N // 10) == 0):
            print(f"[build_SR] Processing image {k+1}/{N} (train index {i})")

        arr = load_image_downsample_gray(image_paths[i], root=root, size=size, verbose=False)
        X[k, :] = arr.flatten()
        R[k, :] = responses[i, :]

    if verbose:
        print(f"[build_SR] Done. X shape: {X.shape}, R shape: {R.shape}")
        print(f"[build_SR] X stats: mean={X.mean():.4f}, std={X.std():.4f}, "
              f"min={X.min():.4f}, max={X.max():.4f}")
        print(f"[build_SR] R stats: mean={R.mean():.4f}, std={R.std():.4f}, "
              f"min={R.min():.4f}, max={R.max():.4f}")

    return X, R, H, W
# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib.patches import Rectangle
def plot_rf_grid(
        rf_per_array, array2area,
        monkey,basename = None,
        shuffle_suffix="true_order", vlim=(-0.5, 0.5),
        save_dir=None, 
        dpi=300
    ):
    """
    Dynamically plots RFs for a variable number of arrays.

    rf_per_array: np.ndarray, shape (N_arrays, H, W)
        RFs per array (each corresponding to 64 electrodes).
    array2area: dict
        Mapping from 1-based array number -> area label.
    """
    n_arrays, H, W = rf_per_array.shape
    n_cols = min(4, n_arrays)  # 4 max per row
    n_rows = math.ceil(n_arrays / n_cols)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.2 * n_rows))
    suptitle = f'{monkey} – GLM-based RFs per array ({shuffle_suffix})'
    fig.suptitle(suptitle, fontsize=18)

    gs = GridSpec(
        n_rows,
        n_cols + 1,  # extra column for colorbar
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.4,
        hspace=0.5
    )

    # color limits
    if vlim is None:
        vmax = np.max(np.abs(rf_per_array))
        vmin = -vmax
    else:
        vmin, vmax = vlim

    
    for idx in range(n_arrays):
        r = idx // n_cols
        c = idx % n_cols
        ax = fig.add_subplot(gs[r, c])

        from matplotlib.patches import Rectangle

        rf = rf_per_array[idx]
        im = ax.imshow(rf, cmap="bwr_r", vmin=vmin, vmax=vmax)
        ax.axis("off")

        # Add a black rectangle that exactly bounds the image
        rect = Rectangle(
            xy=(-0.5, -0.5),  # top-left corner (matches imshow default pixel centers)
            width=W, height=H,
            linewidth=1.5, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)


        array_num = idx + 1  # 1-based
        area = array2area.get(array_num, "unknown")
        ax.set_title(f"Array {array_num} – {area}", fontsize=11, pad=10)

    # colorbar
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
        print(f"[plot_rf_grid_dynamic] Saved PNG to: {png_path}")
        print(f"[plot_rf_grid_dynamic] Saved PDF to: {pdf_path}")

    plt.show()

def compute_array_centers_dva(
        rf_per_array,
        H, W,
        OFFSET_X_PIX_DS, OFFSET_Y_PIX_DS,
        PIX_PER_DEG_THINGS
    ):
    """
    Compute array-level RF centers in DVA space from averaged RF maps.
    """
    n_arrays, _, _ = rf_per_array.shape

    # fovea location in downsampled image pixel coordinates
    cx_pix = (W - 1) / 2.0
    cy_pix = (H - 1) / 2.0
    fovea_x_pix = cx_pix - OFFSET_X_PIX_DS
    fovea_y_pix = cy_pix - OFFSET_Y_PIX_DS

    # pixel grid -> DVA grid
    y_coords, x_coords = np.meshgrid(
        np.arange(H), np.arange(W), indexing="ij"
    )
    x_deg = (x_coords - fovea_x_pix) / PIX_PER_DEG_THINGS
    y_deg = -(y_coords - fovea_y_pix) / PIX_PER_DEG_THINGS

    center_x_deg = np.zeros(n_arrays, dtype=np.float32)
    center_y_deg = np.zeros(n_arrays, dtype=np.float32)

    for a in range(n_arrays):
        rf = rf_per_array[a]
        w = np.abs(rf)
        total = w.sum()
        if total == 0:
            center_x_deg[a] = np.nan
            center_y_deg[a] = np.nan
            continue
        center_x_deg[a] = (w * x_deg).sum() / total
        center_y_deg[a] = (w * y_deg).sum() / total

    return center_x_deg, center_y_deg

def plot_rf_centers_dva(
        center_x_deg, center_y_deg,
        array2area=None,
        title="Utah-array RF centers (GLM)",
        xlim=(-3, 8),
        ylim=(-8, 3)
    ):
    """
    Plot array-level RF centers in DVA space with per-array colors.
    """
    n_arrays = len(center_x_deg)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(n_arrays)]

    for a in range(n_arrays):
        x = center_x_deg[a]
        y = center_y_deg[a]

        if np.isnan(x) or np.isnan(y):
            continue

        if array2area is not None:
            area = array2area.get(a + 1, "unknown")
            if area == "V1":
                marker = "o"
            elif area == "V4":
                marker = "v"
            elif area == "IT":
                marker = "x"
            else:
                marker = "s"
        else:
            marker = "o"

        ax.scatter(
            x, y,
            color=colors[a],
            marker=marker,
            edgecolors="black",
            linewidth=0.8,
            s=80
        )
        # identifier per array
        ax.text(x, y, str(a + 1), fontsize=8, ha="center", va="center", color="white")

    # concentric circles in degrees
    center = (0, 0)
    radius = 8
    num_circles = 4
    for k in range(num_circles):
        circle = patches.Circle(
            center, radius * (k + 1) / num_circles,
            edgecolor="gray", linewidth=0.5, fill=False, linestyle="dashed"
        )
        ax.add_patch(circle)

    ax.axvline(0, color="gray", linestyle="dashed", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="dashed", linewidth=0.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal position (deg)")
    ax.set_ylabel("Vertical position (deg)")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_rf_grid_dva(
        rf_per_array, array2area,
        monkey,
        H, W,
        OFFSET_X_PIX_DS, OFFSET_Y_PIX_DS,
        PIX_PER_DEG_THINGS,
        basename=None,
        shuffle_suffix="true_order", vlim=(-0.5, 0.5),
        save_dir=None,
        dpi=300
    ):
    """
    Plot array-averaged RFs in DVA space, re-centered on fovea using
    downsampled offsets.
    """
    n_arrays, _, _ = rf_per_array.shape
    n_cols = min(4, n_arrays)  # 4 max per row
    n_rows = math.ceil(n_arrays / n_cols)

    fig = plt.figure(figsize=(3.6 * n_cols, 3.6 * n_rows))
    suptitle = f'{monkey} – GLM-based RFs per array (DVA, {shuffle_suffix})'
    fig.suptitle(suptitle, fontsize=18)

    gs = GridSpec(
        n_rows,
        n_cols + 1,  # extra column for colorbar
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.4,
        hspace=0.5
    )

    # color limits
    if vlim is None:
        vmax = np.max(np.abs(rf_per_array))
        vmin = -vmax
    else:
        vmin, vmax = vlim

    # fovea location in downsampled image pixel coordinates
    cx_pix = (W - 1) / 2.0
    cy_pix = (H - 1) / 2.0
    fovea_x_pix = cx_pix - OFFSET_X_PIX_DS
    fovea_y_pix = cy_pix - OFFSET_Y_PIX_DS

    # DVA extents for imshow
    x_deg = (np.arange(W) - fovea_x_pix) / PIX_PER_DEG_THINGS
    y_deg = -(np.arange(H) - fovea_y_pix) / PIX_PER_DEG_THINGS
    extent = [x_deg.min(), x_deg.max(), y_deg.min(), y_deg.max()]

    for idx in range(n_arrays):
        r = idx // n_cols
        c = idx % n_cols
        ax = fig.add_subplot(gs[r, c])

        rf = rf_per_array[idx]
        im = ax.imshow(
            rf,
            cmap="bwr_r",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower"
        )
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.7)
        ax.axvline(0, color="black", linewidth=0.6, alpha=0.7)

        array_num = idx + 1  # 1-based
        area = array2area.get(array_num, "unknown")
        ax.set_title(f"Array {array_num} – {area}", fontsize=11, pad=10)
        ax.set_xlabel("DVA x (deg)")
        ax.set_ylabel("DVA y (deg)")

    # colorbar
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("corr(pixel, MUA)")

    plt.tight_layout(rect=[0, 0, 0.96, 0.95])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"{basename}_DVA.png")
        pdf_path = os.path.join(save_dir, f"{basename}_DVA.pdf")

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_rf_grid_dva] Saved PNG to: {png_path}")
        print(f"[plot_rf_grid_dva] Saved PDF to: {pdf_path}")

    plt.show()
# %%
# Exec prepare_things_obj
train_class, train_local_path, train_things_path = prepare_THINGS_objects(image_MUA_mapping, verbose=False)
## class_refs shape: (22248, 1) dtype: object
## local_path_refs shape: (22248, 1) dtype: object
## things_path_refs shape: (22248, 1) dtype: object
## MUA
with h5py.File(normMUA_path, "r") as f:
    SNR           = np.array(f["SNR"])             # (4, 1024)
    SNR_max       = np.array(f["SNR_max"])         # (1024, 1)
    lats          = np.array(f["lats"])            # (4, 1024)
    oracle        = np.array(f["oracle"])          # (1024, 1)
    reliab        = np.array(f["reliab"])          # (435, 1024)
    tb            = np.array(f["tb"])              # (300, 1)
    test_MUA      = np.array(f["test_MUA"])        # (100, 1024)
    test_MUA_reps = np.array(f["test_MUA_reps"])   # (30, 100, 1024)
    train_MUA     = np.array(f["train_MUA"])       # (22248, 1024)

    # %%

    # PREPARE and ESTIMATE
# specify  images to use for RF estimation
n_total = len(train_things_path)
print(f"[MAIN] Total available training images: {n_total}")

# For speed, potentially use a random subset, REF to n_subset
idxs = np.random.choice(n_total, size=min(n_subset, n_total), replace=False)
print(f"[MAIN] Using {len(idxs)} images for building X.")

# build X (stimuli) and R (responses)
X, R, H, W = build_stimulus_and_response(
    image_paths=train_things_path,
    responses=train_MUA,
    root=object_images_dir,
    size=out_size,
    idxs=idxs,   # or some subset
    verbose=False
)
orig_w, orig_h = get_image_size(train_things_path[idxs[0]], object_images_dir)
print(f"[MAIN] X shape: {X.shape}, R shape: {R.shape}, img_shape: {H, W}")
print(f"[MAIN] First image original size: {orig_w}x{orig_h}")
# --- Monkey N: which Utah array is in which area ---
# WARNING: order is a placeholder – adjust once you know the true array order from TVSD metadata.

if shuffle_mapping:
        N, D = X.shape
        E = R.shape[1]
        # get a random permutation of the N trials
        perm = np.random.permutation(N)        # shape: (N,)
        # shuffle responses relative to stimuli
        R = R[perm, :]    
        order = 'shuffled'
        shuffle_sufix = 'shuffled'
else:            
        order = 'ordered'
        shuffle_sufix = 'true_order'
saving=True
if saving:
        save_dir = join(derivatives_rf_dir, 'linear')
        os.makedirs(save_dir, exist_ok=True)

        data_fname = f"{monkey}_X_R_imgshape{len(idxs)}_{order}.npz"
        save_path = os.path.join(save_dir, data_fname)

        np.savez_compressed(save_path, X=X, R=R, H=H, W=W)
        print(f"[SAVE] Saved stimulus-response data to: {save_path}")


# %%
## LOADING
# Construct load path
loading=False
if loading:
    if shuffle_mapping:
        order = 'shuffled'
    else:
        order = 'ordered'
    len_idxs = 5000 # how many images to include
    load_path = join(derivatives_rf_dir, 'linear', f"{monkey}_X_R_imgshape{len_idxs}_{order}.npz")

    # Load .npz file
    loaded = np.load(load_path)
    X = loaded["X"]
    R = loaded["R"]
    H = int(loaded["H"])
    W = int(loaded["W"])

    print(f"[LOAD] Loaded X shape: {X.shape}, R shape: {R.shape}, img shape: {(H, W)}")

    # %%
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
# ---------- FIT GLM ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
N, D = X_scaled.shape
E = R.shape[1]
assert E % 64 == 0
if E_SUBSET:
     E = E_SUBSET
RFs = np.zeros((E, H * W))
for i in range(E):
        if VERBOSE:
             if i%50 == 0:
                 print(i)
        y = R[:, i]
        #model = Ridge(alpha=0.10)

        # ElasticNet for a balance between sparsity and smoothness
        # l1 ~
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_scaled, y)
        RFs[i] = model.coef_ # shape [n_shanks, 4096]

# %%
# Plotting prepare
# array 
if monkey == 'monkeyN':
    # Nilson
    array2area = {
        1:  'V1', 2: 'V1', 3:  'V1', 4:  'V1',
        5:  'V1',  7:  'V1', 8:  'V1',
        9:  'V4', 10: 'V4', 11: 'V4', 12: 'V4',
        13: 'IT', 14: 'IT', 15: 'IT', 16: 'IT',
        6:  'cut'
    }
elif monkey == 'monkeyF':
    # Finch
    array2area = {
        1:  'V1', 2: 'V1', 3:  'V1', 4:  'V1',
        5:  'V1', 6:  'V1', 7:  'V1', 8:  'V1',
        9:  'V4', 10: 'V4', 11: 'V4', 12: 'V4',
        13: 'IT', 14: 'IT', 15: 'IT', 16: 'IT',
    }
# RFs: (1024, , W') CARTESIAN
n_arrays = E // 64
RFs_reshaped = RFs.reshape(E, 64, 64)  # shape (n_elec, 64, 64)
rf_per_array = RFs_reshaped.reshape(int(E/64), 64, 64, 64).mean(axis=1)  # shape (16, 64, 64, 64)
                                                                                  #^ nr. of electrodes under investigation
# RFs: (1024, , W') Estimate center and SIZE form RF maps
# %% Coordinate grid (pixel indices)
# ---------------------------------------------------------
# Compute RF centers & sizes in PIXEL COORDINATES (RAW)
# ---------------------------------------------------------

y_coords, x_coords = np.meshgrid(
    np.arange(H), np.arange(W), indexing="ij"
)

# Arrays like in THINGS mat file (PIXELS)
all_centrex = np.zeros(E, dtype=np.float32)  # RF center x (pixels)
all_centrey = np.zeros(E, dtype=np.float32)  # RF center y (pixels)
all_szx     = np.zeros(E, dtype=np.float32)  # RF std x (pixels)
all_szy     = np.zeros(E, dtype=np.float32)  # RF std y (pixels)

for i in range(E):
    rf = RFs_reshaped[i]      # (H, W)
    w  = np.abs(rf)           # magnitude as importance
    total = w.sum()

    if total == 0:
        all_centrex[i] = np.nan
        all_centrey[i] = np.nan
        all_szx[i]     = np.nan
        all_szy[i]     = np.nan
        continue

    # --- center of mass (PIXELS, top-left origin) ---
    x_mean = (w * x_coords).sum() / total
    y_mean = (w * y_coords).sum() / total

    # --- spread (PIXELS) ---
    x_var = (w * (x_coords - x_mean) ** 2).sum() / total
    y_var = (w * (y_coords - y_mean) ** 2).sum() / total

    all_centrex[i] = x_mean
    all_centrey[i] = y_mean
    all_szx[i]     = np.sqrt(x_var)
    all_szy[i]     = np.sqrt(y_var)
# ---------------------------------------------------------
# Fovea-centered coordinates (account for stimulus offset)
# ---------------------------------------------------------

# THINGS presentation offset relative to gaze-centered screen center (in *image pixels*)
# IMPORTANT: if you resized stimuli, scale this offset to your current H,W.
# If the original offset was defined in the *same pixel space* as your current H,W, keep as 100.
OFFSET_X_PIX = 100.0
OFFSET_Y_PIX = 100.0

# Downsampled offset in current H,W pixel space (log only, do not recenter yet)
scale_x = W / orig_w
scale_y = H / orig_h
OFFSET_X_PIX_DS = OFFSET_X_PIX * scale_x
OFFSET_Y_PIX_DS = OFFSET_Y_PIX * scale_y
print(f"[OFFSET] Downsampled fixation offset (px): ({OFFSET_X_PIX_DS:.3f}, {OFFSET_Y_PIX_DS:.3f})")

# image center in pixel coordinates (top-left origin)
cx_pix = (W - 1) / 2.0
cy_pix = (H - 1) / 2.0

# fovea location expressed in image pixel coordinates
# (image was shifted right/down by +100 px relative to gaze, so fovea is left/up of image center)
fovea_x_pix = cx_pix - OFFSET_X_PIX
fovea_y_pix = cy_pix - OFFSET_Y_PIX

# Gaze-centered RF centers in degrees:
# +x right, +y up
RFX_fov_deg = (all_centrex - fovea_x_pix) / PIX_PER_DEG_THINGS
RFY_fov_deg = -(all_centrey - fovea_y_pix) / PIX_PER_DEG_THINGS

# sizes are unaffected by centering/flipping (only unit conversion)
STDX_fov_deg = all_szx / PIX_PER_DEG_THINGS
STDY_fov_deg = all_szy / PIX_PER_DEG_THINGS


# ---------------------------------------------------------
# Convert PIXELS → DEGREES (NO center, NO flip)
# ---------------------------------------------------------

RFX_deg  = all_centrex / PIX_PER_DEG_THINGS
RFY_deg  = all_centrey / PIX_PER_DEG_THINGS
STDX_deg = all_szx     / PIX_PER_DEG_THINGS
STDY_deg = all_szy     / PIX_PER_DEG_THINGS


# ---------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------
assert RFs_reshaped.shape == (E, H, W), \
    f"Expected RFs_reshaped {(E,H,W)}, got {RFs_reshaped.shape}"
assert E % 64 == 0, \
    f"E={E} not divisible by 64 (64 electrodes per Utah array)"


# ---------------------------------------------------------
# Save (RAW coordinates)
# ---------------------------------------------------------

rf_save_dir = join(derivatives_rf_dir, "linear")
os.makedirs(rf_save_dir, exist_ok=True)

rf_fname = f"{monkey}_GLM_RFs_{len(idxs)}imgs_{order}_alpha{alpha}_l1r{l1_ratio}.npz"
rf_save_path = os.path.join(rf_save_dir, rf_fname)

np.savez_compressed(
    rf_save_path,

    # electrode-level RFs
    RFs=RFs.astype(np.float32),
    RFs_reshaped=RFs_reshaped.astype(np.float32),

    # array-level RFs
    rf_per_array=rf_per_array.astype(np.float32),
    n_arrays=np.int32(n_arrays),

    # RAW pixel COM + size (top-left origin)
    all_centrex=all_centrex,
    all_centrey=all_centrey,
    all_szx=all_szx,
    all_szy=all_szy,

    # RAW degree version (top-left origin, y-down)
    RFX_deg=RFX_deg,
    RFY_deg=RFY_deg,
    STDX_deg=STDX_deg,
    STDY_deg=STDY_deg,

    # --- NEW: fovea-centered degree version (+x right, +y up) ---
    RFX_fov_deg=RFX_fov_deg.astype(np.float32),
    RFY_fov_deg=RFY_fov_deg.astype(np.float32),
    STDX_fov_deg=STDX_fov_deg.astype(np.float32),
    STDY_fov_deg=STDY_fov_deg.astype(np.float32),

    # --- NEW: fovea definition in pixel coords ---
    OFFSET_X_PIX=np.float32(OFFSET_X_PIX),
    OFFSET_Y_PIX=np.float32(OFFSET_Y_PIX),
    cx_pix=np.float32(cx_pix),
    cy_pix=np.float32(cy_pix),
    fovea_x_pix=np.float32(fovea_x_pix),
    fovea_y_pix=np.float32(fovea_y_pix),

    # metadata
    PIX_PER_DEG_THINGS=np.float32(PIX_PER_DEG_THINGS),
    H=np.int32(H),
    W=np.int32(W),
    E=np.int32(E),
    idxs=idxs.astype(np.int32),
    monkey=str(monkey),
    order=str(order),
    alpha=np.float32(alpha),
    l1_ratio=np.float32(l1_ratio),
)

print(f"[RF SAVE] Saved RFs with BOTH coordinate frames:")
print(f"  RAW (top-left origin, y-down): RFX_deg, RFY_deg")
print(f"  FOVEA-centered (+x right, +y up): RFX_fov_deg, RFY_fov_deg")
print(f"  fovea_pix=({fovea_x_pix:.2f}, {fovea_y_pix:.2f}), offset_pix=({OFFSET_X_PIX}, {OFFSET_Y_PIX})")
print(f"  path: {rf_save_path}")

# %%


# %%
# plot 16 averaged RFs (Utah arrays)
basename, shuffle_suffix = 'test', 'test'

plot_rf_grid(
    rf_per_array, array2area, vlim=vlim,
    monkey="monkeyN", shuffle_suffix=shuffle_suffix,
    save_dir=save_dir, basename=basename
)

# plot 16 averaged RFs (Utah arrays) in DVA space
plot_rf_grid_dva(
    rf_per_array, array2area,
    monkey="monkeyN",
    H=H, W=W,
    OFFSET_X_PIX_DS=OFFSET_X_PIX_DS,
    OFFSET_Y_PIX_DS=OFFSET_Y_PIX_DS,
    PIX_PER_DEG_THINGS=PIX_PER_DEG_THINGS,
    shuffle_suffix=shuffle_suffix,
    vlim=vlim,
    save_dir=save_dir,
    basename=basename
)

# plot array RF centers in DVA space
center_x_deg, center_y_deg = compute_array_centers_dva(
    rf_per_array,
    H=H, W=W,
    OFFSET_X_PIX_DS=OFFSET_X_PIX_DS,
    OFFSET_Y_PIX_DS=OFFSET_Y_PIX_DS,
    PIX_PER_DEG_THINGS=PIX_PER_DEG_THINGS
)
plot_rf_centers_dva(
    center_x_deg, center_y_deg,
    array2area=array2area,
    title="Utah-array RF centers (GLM, DVA)"
)


# %%
