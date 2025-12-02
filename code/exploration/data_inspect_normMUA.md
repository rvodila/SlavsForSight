# THINGS_normMUA (Monkey F) – Variable Overview

This document summarizes the contents of:

`THINGS_normMUA.mat` (Monkey F)

For each variable you’ll find:
- **Name**
- **Dimensions** (in Python / `h5py`)
- **What it represents**
- **Context / how to use it**

---

## 1. Summary Table

| Variable         | Dimensions           | Represents                                                 |
|------------------|----------------------|------------------------------------------------------------|
| `SNR`            | `(4, 1024)`          | Signal-to-noise ratio per **day × electrode**             |
| `SNR_max`        | `(1024, 1)`          | Max SNR per **electrode**                                 |
| `lats`           | `(4, 1024)`          | Response onset latency per **day × electrode** (ms)       |
| `oracle`         | `(1024, 1)`          | Oracle noise ceiling per **electrode**                    |
| `reliab`         | `(435, 1024)`        | Pairwise reliability (correlations) per **electrode**     |
| `tb`             | `(300, 1)`           | Time vector (ms) relative to stimulus onset               |
| `train_MUA`      | `(22248, 1024)`      | Normalized response for **train images**                  |
| `test_MUA`       | `(100, 1024)`        | Normalized response for **test images** (averaged over reps) |
| `test_MUA_reps`  | `(30, 100, 1024)`    | Single-trial normalized responses for **test images**     |

> All dimensions above are given as they appear when loaded with `h5py` in Python.

### Channel mapping:
V1: channels 1–512 in MATLAB → 0–511 in Python
IT: channels 513–832 in MATLAB → 512–831 in Python
V4: channels 833–1024 in MATLAB → 832–1023 in Pytho
---

## 2. Detailed Variable Descriptions

### 2.1 `SNR`

- **Dimensions:** `(4, 1024)`
  - Axis 0: **Day** (4 recording days)
  - Axis 1: **Electrode** (1024 electrodes)
- **Meaning:**  
  Signal-to-noise ratio of the MUA responses, computed separately for each recording day and electrode.
- **Context / use:**
  - Use to **select reliable electrodes** (e.g. threshold on SNR).
  - Can examine how SNR changes across days.

---

### 2.2 `SNR_max`

- **Dimensions:** `(1024, 1)`
  - Axis 0: Electrode
- **Meaning:**  
  Per-electrode maximum SNR, typically based on the preferred test image for that electrode.
- **Context / use:**
  - Single summary SNR value per electrode.
  - Useful as a **quality metric** to select or rank electrodes.

---

### 2.3 `lats`

- **Dimensions:** `(4, 1024)`
  - Axis 0: Day
  - Axis 1: Electrode
- **Meaning:**  
  Latency (in milliseconds) of response onset for each electrode on each day.
- **Context / use:**
  - Characterizes **timing** of neural responses.
  - Can filter electrodes by plausible latency ranges.
  - May be used to compare timing across areas (e.g., V1 vs IT).

---

### 2.4 `oracle`

- **Dimensions:** `(1024, 1)`
  - Axis 0: Electrode
- **Meaning:**  
  Oracle noise ceiling for each electrode—an estimate of the **maximum achievable prediction accuracy** for single-trial responses given the noise level.
- **Context / use:**
  - Commonly used in encoding / decoding models to interpret performance relative to a ceiling.
  - Can be used to **exclude very noisy electrodes** with low ceilings.

---

### 2.5 `reliab`

- **Dimensions:** `(435, 1024)`
  - Axis 0: Pairwise combination index (different pairs of test repetitions)
  - Axis 1: Electrode
- **Meaning:**  
  Reliability of responses per electrode, computed as **pairwise correlations across repetitions** of test images.
- **Context / use:**
  - For each electrode, you can compute:
    ```python
    mean_reliability_per_electrode = reliab.mean(axis=0)
    ```
  - Higher values indicate more stable responses.
  - Another way to **select good electrodes** for modeling.

---

### 2.6 `tb`

- **Dimensions:** `(300, 1)`
  - Axis 0: Time samples
- **Meaning:**  
  Time vector in **milliseconds** relative to stimulus onset.
- **Context / use:**
  - Describes the underlying time axis of the full MUA time courses.
  - The normalized responses in `train_MUA`, `test_MUA`, and `test_MUA_reps` are typically computed by averaging over a specific time window (e.g., 25–125 ms, 50–150 ms, etc.) defined over this `tb` vector.
  - Useful for understanding **which time window** the normalized responses correspond to.

---

## 3. Stimulus-Level MUA Matrices

These matrices give **time-averaged, normalized responses**, not full time courses.

### 3.1 `train_MUA`

- **Dimensions (Python):** `(22248, 1024)`
  - Axis 0: **Train stimulus index** (22,248 training images)
  - Axis 1: **Electrode** (1024)
- **Meaning:**  
  Normalized MUA response for each train image, averaged over the relevant time window.
- **Context / use:**
  - Each **row** corresponds to **one training image**.
  - Each **column** corresponds to an electrode.
  - Used for **model fitting** on a large set of natural images.
  - The row order is aligned with `train_imgs` in `things_imgs.mat` (i.e. `train_MUA[i]` corresponds to `train_imgs[i]`).

---

### 3.2 `test_MUA`

- **Dimensions (Python):** `(100, 1024)`
  - Axis 0: **Test stimulus index** (100 test images)
  - Axis 1: **Electrode**
- **Meaning:**  
  Normalized MUA response for each test image, **averaged across repetitions**, within the defined time window.
- **Context / use:**
  - Each **row** corresponds to **one test image**.
  - Often used for **evaluation / testing** after fitting models on `train_MUA`.
  - The row order is aligned with `test_imgs` in `things_imgs.mat` (i.e. `test_MUA[j]` corresponds to `test_imgs[j]`).

---

### 3.3 `test_MUA_reps`

- **Dimensions (Python):** `(30, 100, 1024)`
  - Axis 0: **Repetition** (30 repetitions per test image)
  - Axis 1: **Test stimulus index** (matching `test_MUA`)
  - Axis 2: **Electrode**
- **Meaning:**  
  Single-trial normalized MUA responses for test images: one entry per repetition, image, and electrode.
- **Context / use:**
  - Allows analysis of **trial-to-trial variability** and **single-trial decoding**.
  - For a given test image `j` and electrode `e`, you can extract all repetitions:
    ```python
    # all repetitions for test image j on electrode e
    reps_j_e = test_MUA_reps[:, j, e]  # shape: (30,)
    ```
  - If you prefer electrodes as the first dimension:
    ```python
    test_MUA_reps_e_first = np.moveaxis(test_MUA_reps, 2, 0)  # (1024, 30, 100)
    ```

---

## 4. High-Level Context

- The variables in `THINGS_normMUA.mat` provide:
  - **Pre-processed, normalized, and time-averaged responses** at the **stimulus level**.
  - **Quality and reliability metrics** at the electrode (and day) level.
- To link these responses to:
  - **Specific trials** and
  - **Actual image files / THINGS IDs**,  
  you also use:
  - `THINGS_MUA_trials.mat` (trial-wise metadata, including `train_idx` and `test_idx`)  
  - `_logs/things_imgs.mat` (mapping from those indices to THINGS image IDs / filenames).

This file (`THINGS_normMUA.mat`) is therefore best thought of as a **clean, analysis-ready summary** of MUA responses and their quality metrics, organized by **stimulus** (train/test) and **electrode**, with additional temporal and reliability context.
