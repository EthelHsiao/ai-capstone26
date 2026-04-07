# HW2 — 3D Scene Reconstruction

## Environment

Use the **same `habitat` conda environment from HW0** — no second environment needed.

```bash
conda activate habitat
```

All required packages (`open3d`, `scipy`, `numpy==1.26.4`, `opencv-python`, `habitat-sim`) are already installed from HW0's `requirements.txt`.

> **Why not two environments?**
> The GLFW conflict only occurs when `habitat_sim` and `open3d` are imported in the **same script**.
> Since `load.py` only uses `habitat_sim` and `reconstruct.py` only uses `open3d`, running them separately in the same environment is safe.

---

## Step 1 — Download the Replica Dataset

> Skip this step if you already downloaded `apartment_0` in HW0.

Place the dataset under `hw2/replica_v1/` so the folder looks like:

```
hw2/
└── replica_v1/
    └── apartment_0/
        ├── habitat/
        │   └── mesh_semantic.ply   ← this file is required
        ├── textures/
        └── ...
```

### Download command

```bash
cd hw2
gdown https://drive.google.com/uc?id=1zHA2AYRtJOmlRaHNuXOvC_OaVxHe56M4 -O apartment_0.zip
unzip apartment_0.zip -d replica_v1/
rm apartment_0.zip
```

If `gdown` is not installed:

```bash
pip install gdown
```

---

## Step 2 — Collect RGB-D Data

Run `load.py` to manually explore the apartment and save sensor data.

```bash
# Collect first floor
python load.py -f 1

# Collect second floor
python load.py -f 2
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `w` | Move forward |
| `a` | Turn left |
| `d` | Turn right |
| `f` | Finish and save |

### What gets saved

Collected data is saved to `data_collection/first_floor/` (or `second_floor/`):

```
data_collection/
└── first_floor/
    ├── rgb/          1.png, 2.png, ...    (colour images)
    ├── depth/        1.png, 2.png, ...    (uint16 PNG, depth in mm)
    ├── semantic/     1.png, 2.png, ...    (semantic segmentation)
    └── GT_pose.npy                        (N × 7: x y z qw qx qy qz)
```

### Tips for good data collection

- **Cover the entire floor** — walk into every room and corner.
- **Aim for ~150 frames per floor** — more frames = better coverage, but slower reconstruction.
- **Move slowly** — large jumps between frames make ICP alignment harder.
- **Turn in place occasionally** — helps capture 360° geometry of each spot.
- **Avoid pointing the camera at blank walls only** — featureless surfaces confuse RANSAC.

> ⚠️ **Warning:** Running `load.py` again will **delete** the existing data folder and start fresh.

---

## Step 3 — 3D Reconstruction

```bash
# Required: Open3D ICP
python reconstruct.py -f 1 -v open3d

# Bonus: custom ICP implementation
python reconstruct.py -f 1 -v my_icp

# Second floor
python reconstruct.py -f 2 -v open3d
```

### Output

- A 3D viewer window opens showing:
  - Reconstructed point cloud (ceiling removed)
  - **Red line** — estimated camera trajectory (from ICP)
  - **Black line** — ground truth camera trajectory (`GT_pose.npy`)
- `Mean L2 distance: X.XXXX m` is printed to the terminal.

---

## Tuning Guide — What to Change When Results Look Wrong

### 1. Ceiling not fully removed

```python
# reconstruct.py — near the bottom of __main__
ceil_above = 1.0    # ← increase this value (e.g. 1.5 or 2.0)
```

In the camera's OpenCV frame Y-axis points downward, so the ceiling sits at
large **negative** Y. Increasing `ceil_above` removes more of the upper region.

---

### 2. Reconstruction is noisy / misaligned

```python
# reconstruct.py — top of reconstruct()
voxel_size = 0.02   # ← try 0.03 or 0.05 for coarser but faster results
                    #   try 0.01 for finer but slower results
```

| `voxel_size` | Effect |
|---|---|
| smaller (0.01) | More detail, slower, easier to misalign |
| larger (0.05)  | Less detail, faster, more robust to noisy data |

---

### 3. RANSAC / ICP threshold

```python
# reconstruct.py
icp_thresh = voxel_size * 0.4   # ← increase multiplier if ICP diverges
                                 #   e.g. voxel_size * 0.6
```

Inside `execute_global_registration`:
```python
distance_threshold = voxel_size * 1.5   # ← increase if RANSAC finds few inliers
```

---

### 4. Custom ICP runs too slow

```python
# my_local_icp_algorithm()
max_iters = 50    # ← reduce to 20–30 to speed up (less accurate)
threshold = voxel_size * 1.5   # ← increase to match more points (faster convergence)
```

---

### 5. Red trajectory drifts far from black (high L2 error)

Possible causes and fixes:

| Cause | Fix |
|---|---|
| Too few frames collected | Re-collect with more frames (~150+) |
| Frames have large gaps (fast movement) | Move slower / forward step by step |
| Featureless scene areas | Include more textured areas, turn more |
| `voxel_size` too large | Decrease `voxel_size` |

---

## File Structure

```
hw2/
├── load.py              # Task 1 — data collection (habitat_sim)
├── reconstruct.py       # Task 2 & 3 — reconstruction (open3d + custom ICP)
├── report.pdf           # Detailed report (English)
├── README.md            # This file
└── replica_v1/
    └── apartment_0/     # Replica dataset scene
```

---

## Troubleshooting

### Segmentation fault on `Vector3dVector`
```bash
pip install numpy==1.26.4
```

### `ImportError: libOpen3D.so` / OpenMP missing (Linux)
```bash
sudo apt-get install libomp-dev
```

### `open3d` fails to load (macOS)
```bash
brew install libomp
```

### `ModuleNotFoundError: scipy`
```bash
pip install scipy
```
