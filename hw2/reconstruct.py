import numpy as np
import open3d as o3d
import argparse
import os
import copy
import cv2
import time
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics  (same as HW1: 512×512, FOV=90°)
#   f  = (W/2) / tan(FOV/2) = 256 / tan(45°) = 256 px
#   cx = cy = 256  (principal point at image centre)
# ─────────────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 512, 512
FOV_DEG       = 90.0
fx = fy = (WIDTH / 2) / np.tan(np.radians(FOV_DEG / 2))   # 256.0
cx = WIDTH  / 2   # 256.0
cy = HEIGHT / 2   # 256.0

# Habitat uses OpenGL convention (Y-up, Z-backward).
# Our depth un-projection works in OpenCV convention (Y-down, Z-forward).
# GL2CV flips Y and Z to switch between the two.
GL2CV = np.diag([1.0, -1.0, -1.0])


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-1 : Un-project depth image → point cloud
# ─────────────────────────────────────────────────────────────────────────────
def depth_image_to_point_cloud(rgb, depth):
    """
    Convert an RGB image and a depth image into a coloured 3-D point cloud.

    The implementation follows the pinhole camera model (OpenCV convention):
        Z  = depth[v, u]             (metres)
        X  = (u - cx) * Z / fx
        Y  = (v - cy) * Z / fy

    This is equivalent to:  [X, Y, Z]^T  =  Z * K_inv @ [u, v, 1]^T
    where K_inv is the inverse of the 3×3 intrinsic matrix.

    Args:
        rgb   : (H, W, 3)  uint8  RGB image
        depth : (H, W)     float32  depth in METRES

    Returns:
        open3d.geometry.PointCloud  in the camera's OpenCV frame
        (X right, Y down, Z forward)

    Note: Open3D functions are intentionally NOT used here.
    """
    H, W = depth.shape

    # --- pixel coordinate grids ---
    u_grid = np.arange(W, dtype=np.float64)
    v_grid = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid)   # both (H, W)

    Z = depth.astype(np.float64)           # (H, W)  metres

    # --- pinhole un-projection (manual, no Open3D) ---
    X = (uu - cx) * Z / fx                 # (H, W)
    Y = (vv - cy) * Z / fy                 # (H, W)

    # --- filter pixels with invalid (zero / near-zero) depth ---
    valid   = Z > 0.01                     # (H, W) boolean mask

    points  = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)   # (N, 3)
    colors  = rgb[valid].astype(np.float64) / 255.0               # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-2 : Voxel down-sampling + normal estimation + FPFH features
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_point_cloud(pcd, voxel_size):
    """
    Down-sample a point cloud with a voxel grid, estimate surface normals,
    and compute FPFH feature descriptors needed for RANSAC.

    Returns:
        pcd_down  : voxel-downsampled PointCloud
        pcd_fpfh  : FPFH feature for every point in pcd_down
    """
    # 1. voxel down-sample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 2. estimate normals (required for Point-to-Plane ICP and FPFH)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # 3. compute FPFH feature (33-dim histogram describing local geometry)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-3 : Global registration with RANSAC
# ─────────────────────────────────────────────────────────────────────────────
def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh, voxel_size):
    """
    Estimate a coarse initial alignment between two point clouds using
    RANSAC + FPFH feature matching.

    RANSAC iteratively:
      1. Randomly samples 4 point-pairs whose FPFH descriptors are similar.
      2. Computes a candidate rigid transform.
      3. Counts inliers (correspondences within distance_threshold).
      4. Keeps the transform with the most inliers.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        # 4,000,000 iterations is extremely slow on laptops; this keeps alignment
        # quality reasonable while making runtime practical for homework-scale runs.
        # TEST
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40_000, 200),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-4 : Local refinement with Open3D ICP  (Required)
# ─────────────────────────────────────────────────────────────────────────────
def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    Refine the coarse alignment produced by RANSAC using Point-to-Plane ICP.

    Point-to-Plane ICP minimises the sum of squared distances from each
    source point to the *tangent plane* at its nearest target point.
    This converges faster than Point-to-Point ICP.

    Args:
        source_down : source PointCloud (with normals)
        target_down : target PointCloud (with normals)
        trans_init  : 4×4 initial transformation matrix (from RANSAC)
        threshold   : max correspondence distance (metres)

    Returns:
        Open3D RegistrationResult  (.transformation = refined 4×4 matrix)
    """
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-4 Bonus : Custom ICP implementation
# ─────────────────────────────────────────────────────────────────────────────

def _svd_rigid_transform(src, tgt):
    """
    Compute the best-fit rigid transform R, t that maps src → tgt.

    Algorithm (Arun et al., 1987):
      1. Compute centroids and centre both sets.
      2. Form cross-covariance matrix H = src_c^T @ tgt_c.
      3. SVD decompose H = U Σ V^T.
      4. R = V U^T   (corrected for reflections).
      5. t = tgt_mean - R @ src_mean.

    Args:
        src : (N, 3) source points
        tgt : (N, 3) corresponding target points

    Returns:
        T : (4, 4) rigid transform matrix
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = src_c.T @ tgt_c                        # (3, 3)
    U, _, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T
    # Handle reflection (det should be +1 for a proper rotation)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    t_vec = tgt_mean - R_mat @ src_mean

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3]  = t_vec
    return T


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Custom Point-to-Point ICP using scipy KDTree for fast nearest-neighbour
    search and SVD for closed-form transformation estimation.

    Performance tricks used:
      * scipy KDTree.query() is vectorised (all points in one call) → fast.
      * Distance threshold prunes bad correspondences before SVD.
      * Early stopping when RMS improvement < tol.
      * Max 50 iterations is sufficient for pre-aligned (post-RANSAC) inputs.

    Args:
        source_down : source PointCloud
        target_down : target PointCloud
        trans_init  : (4, 4) initial transform from RANSAC
        voxel_size  : used to derive the correspondence distance threshold

    Returns:
        Object with attribute .transformation  (4×4 accumulated transform)
    """
    # TEST:Params
    threshold  = voxel_size * 2
    max_iters  = 20
    tol        = 1e-4

    src_pts = np.asarray(source_down.points, dtype=np.float64)   # (N, 3)
    tgt_pts = np.asarray(target_down.points, dtype=np.float64)   # (M, 3)

    # --- apply initial transform from RANSAC ---
    T_cum = trans_init.astype(np.float64).copy()
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    src_t = (T_cum @ src_h.T).T[:, :3]        # (N, 3) in target's frame

    # --- build KD-Tree on target (built once, reused every iteration) ---
    kd_tree = KDTree(tgt_pts)

    prev_rms = np.inf

    for _ in range(max_iters):
        # Step 1 – vectorised nearest-neighbour search
        dist, idx = kd_tree.query(src_t, workers=-1)  # TEST:用所有 CPU 核心     # dist: (N,), idx: (N,)
        valid = dist < threshold
        if valid.sum() < 6:
            break                               # too few correspondences

        src_m = src_t[valid]
        tgt_m = tgt_pts[idx[valid]]

        # Step 2 – convergence check
        rms = dist[valid].mean()
        if abs(prev_rms - rms) < tol:
            break
        prev_rms = rms

        # Step 3 – SVD-based rigid transform estimation
        T_step = _svd_rigid_transform(src_m, tgt_m)

        # Step 4 – apply incremental transform
        T_cum = T_step @ T_cum
        src_t = (T_step[:3, :3] @ src_t.T).T + T_step[:3, 3]

    class _ICPResult:
        def __init__(self, T):
            self.transformation = T

    return _ICPResult(T_cum)


# ─────────────────────────────────────────────────────────────────────────────
# GT-pose helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(qw, qx, qy, qz):
    """Quaternion [qw, qx, qy, qz] → 3×3 rotation matrix."""
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _gt_positions_in_icp_frame(gt_poses):
    """
    Convert all GT world positions (Habitat / OpenGL convention) into the
    first camera's OpenCV frame so they can be directly compared with the
    ICP-accumulated positions (pred_cam_pos).

    Steps:
        1. Build T0 (cam-0 → world, OpenGL) from the first GT pose.
        2. Invert T0 to get world → cam-0 (OpenGL).
        3. Apply GL2CV to convert OpenGL → OpenCV.

    Args:
        gt_poses : (N, 7) array  [x, y, z, qw, qx, qy, qz]

    Returns:
        (N, 3) positions in cam-0 OpenCV frame
    """
    x0, y0, z0, qw0, qx0, qy0, qz0 = gt_poses[0]
    R0  = _quat_to_rot(qw0, qx0, qy0, qz0)      # cam-0 → world rotation
    t0  = np.array([x0, y0, z0])

    # T0 : cam-0 OpenGL → world
    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3]  = t0
    T0_inv = np.linalg.inv(T0)                   # world → cam-0 OpenGL

    positions = []
    for pose in gt_poses:
        p_world = np.append(pose[:3], 1.0)
        p_cam0_gl = (T0_inv @ p_world)[:3]        # in cam-0 OpenGL frame
        p_cam0_cv = GL2CV @ p_cam0_gl             # flip Y, Z → OpenCV frame
        positions.append(p_cam0_cv)

    return np.array(positions)                    # (N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 + 3 : Full reconstruction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(args):
    """
    Full pipeline:
        load frames → un-project depth → downsample → RANSAC → ICP
        → accumulate transforms → merge point clouds

    Returns:
        result_pcd   : merged + downsampled global PointCloud (ICP / cam-0 frame)
        pred_cam_pos : (N, 3) estimated camera positions in the same frame
    """
    data_root  = args.data_root
    voxel_size = 0.05          # 5 cm – good balance of speed and accuracy
    icp_thresh = voxel_size * 1.5

    depth_dir = os.path.join(data_root, 'depth')
    rgb_dir   = os.path.join(data_root, 'rgb')
    gt_poses  = np.load(os.path.join(data_root, 'GT_pose.npy'))   # (N, 7)
    n_frames  = len(gt_poses)

    print(f"[reconstruct] version={args.version}, frames={n_frames}, "
          f"voxel_size={voxel_size}")

    def load_frame(idx_1based):
        """Load RGB (H,W,3 uint8) and depth (H,W float32 metres) for file <idx>.png."""
        rgb_path   = os.path.join(rgb_dir,   f'{idx_1based}.png')
        depth_path = os.path.join(depth_dir, f'{idx_1based}.png')

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Depth saved as uint16 millimetres → divide by depth_scale=1000 → metres
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m   = depth_raw / 1000.0
        return rgb, depth_m

    # ── initialise ────────────────────────────────────────────────────────────
    T_cum        = np.eye(4)                      # frame 0 = reference (identity)
    pred_cam_pos = [T_cum[:3, 3].copy()]          # camera 0 at origin
    result_pcd   = o3d.geometry.PointCloud()

    # ── frame 0 ───────────────────────────────────────────────────────────────
    rgb0, depth0 = load_frame(1)
    pcd_prev = depth_image_to_point_cloud(rgb0, depth0)
    prev_down, prev_fpfh = preprocess_point_cloud(pcd_prev, voxel_size)
    result_pcd += copy.deepcopy(pcd_prev)         # add frame 0 (already at origin)

    # ── frames 1 … N-1 ────────────────────────────────────────────────────────
    for i in range(1, n_frames):
        print(f"\r  frame {i}/{n_frames-1}", end='', flush=True)

        rgb_i, depth_i = load_frame(i + 1)        # files are 1-indexed
        pcd_curr = depth_image_to_point_cloud(rgb_i, depth_i)

        # ─ voxel downsample + FPFH ────────────────────────────────────────────
        src_down, src_fpfh = preprocess_point_cloud(pcd_curr, voxel_size)
        tgt_down, tgt_fpfh = prev_down, prev_fpfh

        # ─ global registration (RANSAC) ────────────────────────────────────────
        # TEST
        # 每兩幀跑一次 RANSAC，下一幀重用同一個結果
        # if i % 2 == 1:  # 奇數幀跑 RANSAC
        #     ransac = execute_global_registration(src_down, tgt_down,
        #                                  src_fpfh, tgt_fpfh,
        #                                  voxel_size)
        # trans_init = ransac.transformation  # 奇數偶數幀都用同一個
        ransac = execute_global_registration(src_down, tgt_down,
                                     src_fpfh, tgt_fpfh,
                                     voxel_size)

        # ─ local registration (ICP) ────────────────────────────────────────────
        if args.version == 'open3d':
            icp_res = local_icp_algorithm(src_down, tgt_down,
                                          ransac.transformation, icp_thresh)
        else:   # 'my_icp'
            icp_res = my_local_icp_algorithm(src_down, tgt_down,
                                             ransac.transformation, voxel_size)

        # T_icp : curr → prev  (maps frame i's points into frame i-1's coords)
        T_icp = icp_res.transformation

        # Accumulate:  T_{0←i} = T_{0←(i-1)} @ T_{(i-1)←i}
        T_cum = T_cum @ T_icp

        # Camera i's centre ([0,0,0] in its own frame) mapped to frame 0:
        pred_cam_pos.append(T_cum[:3, 3].copy())

        # Merge current frame into global point cloud
        pcd_curr_global = copy.deepcopy(pcd_curr)
        pcd_curr_global.transform(T_cum)
        result_pcd += pcd_curr_global

        pcd_prev = pcd_curr   # advance (kept in camera frame for next ICP)
        prev_down, prev_fpfh = src_down, src_fpfh

    print()

    # ── final downsample (memory & display efficiency) ────────────────────────
    result_pcd = result_pcd.voxel_down_sample(voxel_size * 2) # TEST: draw 
    # result_pcd = result_pcd.voxel_down_sample(0.02) 
    pred_cam_pos = np.array(pred_cam_pos)   # (N, 3)
    return result_pcd, pred_cam_pos


def run_with_timing(func, *args, **kwargs):
    """Run a function and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor',   type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d',
                        help='open3d  or  my_icp')
    parser.add_argument('--data_root', type=str,
                        default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = 'data_collection/first_floor/'
    elif args.floor == 2:
        args.data_root = 'data_collection/second_floor/'

    # ── run reconstruction with timing ───────────────────────────────────────
    (result_pcd, pred_cam_pos), elapsed_sec = run_with_timing(reconstruct, args)
    print(f"Reconstruction runtime: {elapsed_sec:.2f} s ({elapsed_sec / 60.0:.2f} min)")

    # ── load GT and convert to ICP frame ─────────────────────────────────────
    gt_poses = np.load(os.path.join(args.data_root, 'GT_pose.npy'))
    gt_pos   = _gt_positions_in_icp_frame(gt_poses)    # (N, 3) in cam-0 CV frame
    n        = len(pred_cam_pos)
    gt_pos   = gt_pos[:n]

    # ── Mean L2 distance ──────────────────────────────────────────────────────
    l2_per_frame = np.linalg.norm(pred_cam_pos - gt_pos, axis=1)  # (N,)
    mean_l2      = l2_per_frame.mean()
    print(f"Mean L2 distance: {mean_l2:.4f} m")

    # ── Remove ceiling ────────────────────────────────────────────────────────
    # In the camera's OpenCV frame: Y is DOWN → ceiling is at NEGATIVE Y.
    # Points more than `ceil_above` metres above the camera are discarded.
    # Tune this value if the ceiling is not fully removed.
    ceil_above = 0.6      # metres above camera level → Y < -ceil_above
    pts  = np.asarray(result_pcd.points)
    cols = np.asarray(result_pcd.colors)
    mask = pts[:, 1] > -ceil_above
    trimmed = o3d.geometry.PointCloud()
    trimmed.points = o3d.utility.Vector3dVector(pts[mask])
    trimmed.colors = o3d.utility.Vector3dVector(cols[mask])

    # ── Build trajectory line sets ────────────────────────────────────────────
    def make_lineset(positions, color_rgb):
        ls  = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(positions)
        ls.lines  = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(positions) - 1)]
        )
        ls.colors = o3d.utility.Vector3dVector(
            [color_rgb] * (len(positions) - 1)
        )
        return ls

    est_traj = make_lineset(pred_cam_pos, [1, 0, 0])   # Red  – estimated
    gt_traj  = make_lineset(gt_pos,       [0, 0, 0])   # Black – ground truth

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("Opening 3D viewer …  (close the window to exit)")
    o3d.visualization.draw_geometries(
        [trimmed, est_traj, gt_traj],
        window_name='3D Scene Reconstruction',
        width=1280, height=720,
    )
