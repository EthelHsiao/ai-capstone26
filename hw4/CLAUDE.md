# HW4: Robot Manipulation Framework — CLAUDE.md

> **Course:** AI Capstone (NYCU 2026 Spring)
> **Deadline:** 2026/05/12 23:59
> **Submission:** `{STUDENT_ID}_hw4.zip` → New E3 System

---

## Overview

Implement **Forward Kinematics (FK)** and **Inverse Kinematics (IK)** for a **UR5 6-DOF robot arm** in Nvidia Isaac Sim. Two coding tasks + a written report.

## Project Structure

```
hw4/
├── fk.py           # Task 1: implement your_fk()
├── ik.py           # Task 2: implement your_ik()
├── test_case/      # public test cases for verification
├── README.md
├── pyproject.toml  # deps: isaacsim[all,extscache]==5.1.0, scipy
├── Dockerfile
├── docker-compose.yaml
├── docker.py       # CLI: run-fk / run-ik (--headless)
└── entrypoint.sh
```

## Submission Format

```
{student_id}_hw4.zip
├── fk.py           # your_fk() implementation
├── ik.py           # your_ik() implementation
├── report.pdf      # written in English
└── (other files if modified)
```

**Penalties:** wrong format → -10 pts, late → -20 pts/day, plagiarism → 0, report not in English → 0.

---

## Grading Breakdown (Total: 100 + 5 bonus)

| Component             | Weight |
|----------------------|--------|
| Task 1 test cases (FK) | 30%  |
| Task 2 test cases (IK) | 40%  |
| Report Q1 (about FK)   | 15%  |
| Report Q2 (about IK)   | 15%  |
| Bonus (other IK methods)| 5%  |

### Scoring Thresholds (from code)

- FK pose error threshold: **0.005**
- Jacobian error threshold: **0.05**
- FK score max: 10, Jacobian score max: 10, total Task1 max: 20
- Task 2 scored by mean error and error count over 100 test cases per file

---

## Task 1: Forward Kinematics (`fk.py` → `your_fk()`)

### Function Signature

```python
def your_fk(DH_params: dict, q, base_pos) -> tuple[np.ndarray, np.ndarray]:
```

### Input
- `DH_params`: list of 6 dicts, each with keys `a`, `d`, `alpha` (classic DH convention)
- `q`: 6D array of joint angles in radians, range [-π, π]
- `base_pos`: robot base position [x, y, z] in world frame

### Output
- `pose_7d`: np.ndarray [x, y, z, qx, qy, qz, qw] — end-effector pose (quaternion in xyzw format)
- `jacobian`: np.ndarray (6×6) — geometric Jacobian in base frame

### DH Parameters (Classic Convention) — USE THESE, NOT OFFICIAL UR5 SPEC

```python
dh_params = [
    {'a':  0,      'd': 0.0892,  'alpha':  π/2  },  # joint1
    {'a': -0.425,  'd': 0,       'alpha':  0    },  # joint2
    {'a': -0.392,  'd': 0,       'alpha':  0    },  # joint3
    {'a':  0,      'd': 0.1093,  'alpha':  π/2  },  # joint4
    {'a':  0,      'd': 0.09475, 'alpha': -π/2  },  # joint5
    {'a':  0,      'd': 0.2023,  'alpha':  0    },  # joint6
]
```

### Implementation Notes

1. **Classic DH Transform** for joint i:
   ```
   T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)
   ```
   As a 4×4 matrix:
   ```
   [cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ)]
   [sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ)]
   [  0        sin(α)         cos(α)            d    ]
   [  0          0              0               1    ]
   ```

2. **Chain multiplication**: `T_0^6 = T_base · T_1 · T_2 · T_3 · T_4 · T_5 · T_6`

3. **Geometric Jacobian** (for revolute joints):
   - Column i: `Jv_i = z_i × (p_end - p_i)`, `Jw_i = z_i`
   - Where `z_i` = z-axis of frame i (3rd column of rotation), `p_i` = origin of frame i

4. **Post-processing (DO NOT TOUCH)**:
   ```python
   adjustment = np.asarray([[ 0, -1,  0],
                            [ 0,  0,  0],
                            [ 0,  0, -1]])
   A[:3, :3] = A[:3,:3] @ adjustment
   pose_7d = get_pose_from_matrix(A, 7)
   ```

5. **Allowed libraries**: numpy, scipy.spatial.transform, quaternion, numba — NO Isaac Sim / pybullet APIs

### Helper Functions Available in fk.py

- `get_matrix_from_pose(pose)` → 4×4 homogeneous transform (6D rotvec or 7D quat)
- `get_pose_from_matrix(matrix, pose_size=7)` → 6D or 7D pose vector
- `cross(a, b)` → np.cross wrapper

### Verification

```bash
python fk.py                        # with GUI
python fk.py --headless             # headless
python3 docker.py run-fk --headless # via Docker
```

Expected perfect output:
```
- Your Score Of Forward Kinematic : 5.000 / 5.000, Error Count :    0 /  100
- Your Score Of Jacobian Matrix   : 5.000 / 5.000, Error Count :    0 /  100
(× 2 test files = 20.000 / 20.000 total)
```

---

## Task 2: Inverse Kinematics (`ik.py` → `your_ik()`)

### Function Signature

```python
def your_ik(new_pose, base_pos, q_init=None) -> list:
```

### Input
- `new_pose`: target 7D pose [x, y, z, qx, qy, qz, qw]
- `base_pos`: robot base position [x, y, z]
- `q_init`: initial 6D joint angles (optional, will use default if None)

### Output
- `list` of 6 joint angles (radians) that achieve the target pose

### Joint Limits

```python
joint_limits = np.asarray([
    [-3*np.pi/2, -np.pi/2],  # joint1
    [-2.3562, -1],            # joint2
    [-17, 17],                # joint3
    [-17, 17],                # joint4
    [-17, 17],                # joint5
    [-17, 17],                # joint6
])
```

### Required Method: Iterative IK with Jacobian Pseudo-Inverse

1. Evaluate current pose & Jacobian via `your_fk()`
2. Compute 6D error `[position_error(3), orientation_error(3)]`
   - Position error: `p_target - p_current`
   - Orientation error: from relative rotation `R_target @ R_current.T` → axis-angle
3. Compute `Δq = J⁺ · error` (pseudo-inverse: `J⁺ = np.linalg.pinv(J)`)
4. Apply step size, clip by joint limits
5. Iterate until error norm < threshold

### Key Implementation Tips

- Use `your_fk()` inside the IK loop
- Be careful with orientation error computation (axis-angle from rotation matrix)
- Tune hyperparameters: step rate (learning rate), max iterations, convergence threshold
- Clip joints to `joint_limits` each iteration

### Verification

```bash
python ik.py                        # with GUI
python ik.py --headless             # headless
python3 docker.py run-ik --headless # via Docker
```

Expected perfect output:
```
- Mean Error : 0.001048
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000
(× 2 test files = 40.000 / 40.000 total)
```

### Bonus (+5%): Implement alternative IK methods

e.g., Damped Least Squares (DLS), Jacobian Transpose, CCD, FABRIK — compare results with pseudo-inverse.

---

## Report Questions

### 1. About Task 1 (15%)

- **1.1 (3%)** Explain your `your_fk()` implementation (can include code screenshots)
- **1.2 (2%)** Difference between classic D-H convention vs Craig's convention (Modified D-H)
- **1.3 (10%)** Complete the D-H table (7 rows: i=1..7) with columns: `i | d | α (rad) | a | θ_i (rad)`
  - The coordinate frames diagram is provided in the spec (page 10)
  - Note: table has 7 rows (includes a tool frame), θ columns are θ₁..θ₇

### 2. About Task 2 (15% + 5% bonus)

- **2.1 (10%)** Explain your `your_ik()` implementation (can include code screenshots)
- **2.2 (5%)** Problems encountered and solutions
- **2.3 (5% bonus)** Other IK methods implemented + comparison

---

## Environment Setup

### Option 1: Local with uv (Ubuntu recommended)

```bash
cd hw4
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
python fk.py
python ik.py
```

### Option 2: Docker

```bash
cd hw4
python3 docker.py run-fk --headless
python3 docker.py run-ik --headless
```

### Option 3: Glows.AI (handles setup automatically)

### System Requirements
- GPU: RTX 4080+ (16GB VRAM minimum)
- RAM: 32GB minimum
- Python 3.11
- Ubuntu 22.04 recommended

---

## References

- Isaac Sim docs: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
- UR5 DH parameters (official, slightly different from HW): https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
- Jacobian IK methods: https://homes.cs.washington.edu/~todorov/courses/cseP590/06_JacobianMethods.pdf
- Jacobian tutorial: https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/

---

## Quick Implementation Checklist

- [ ] **Task 1**: Implement classic DH transform matrix function
- [ ] **Task 1**: Chain-multiply transforms T_base · T_1 · ... · T_6
- [ ] **Task 1**: Store z_i and p_i at each joint for Jacobian
- [ ] **Task 1**: Build 6×6 Jacobian: `Jv_i = z_i × (p_end - p_i)`, `Jw_i = z_i`
- [ ] **Task 1**: Verify with `python fk.py --headless` → 20/20
- [ ] **Task 2**: Implement IK loop calling `your_fk()` each iteration
- [ ] **Task 2**: Compute position error (simple subtraction)
- [ ] **Task 2**: Compute orientation error (rotation matrix → axis-angle)
- [ ] **Task 2**: Apply pseudo-inverse: `Δq = pinv(J) @ error`
- [ ] **Task 2**: Apply step size + joint limit clipping
- [ ] **Task 2**: Tune convergence threshold and max iterations
- [ ] **Task 2**: Verify with `python ik.py --headless` → 40/40
- [ ] **Report**: Write all sections in English, export as PDF
- [ ] **Submission**: Zip as `{STUDENT_ID}_hw4.zip` with correct structure