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

    Args:
        rgb   : (H, W, 3)  uint8  RGB image
        depth : (H, W)     float32  depth in METRES

    Returns:
        open3d.geometry.PointCloud  in the camera's OpenCV frame
    """
    H, W = depth.shape

    u_grid = np.arange(W, dtype=np.float64)
    v_grid = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid)

    Z = depth.astype(np.float64)

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    valid = Z > 0.01

    points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    colors = rgb[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-2 : Voxel down-sampling + normal estimation + FPFH features
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_point_cloud(pcd, voxel_size):
    """
    Down-sample, estimate normals, compute FPFH features.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

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
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40_000, 200),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-4 : Local refinement with Open3D ICP  (Required)
# ─────────────────────────────────────────────────────────────────────────────
def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task 2-4 Bonus : Custom ICP — Point-to-Point (SVD)
# ─────────────────────────────────────────────────────────────────────────────

def _svd_rigid_transform(src, tgt):
    """
    Compute the best-fit rigid transform R, t that maps src → tgt
    using SVD (Arun et al., 1987).
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    t_vec = tgt_mean - R_mat @ src_mean

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3]  = t_vec
    return T


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size,
                           icp_method='point_to_point'):
    """
    Custom ICP implementation.

    Args:
        icp_method : 'point_to_point' or 'point_to_plane'
    """
    if icp_method == 'point_to_plane':
        return _my_icp_point_to_plane(source_down, target_down, trans_init, voxel_size)
    else:
        return _my_icp_point_to_point(source_down, target_down, trans_init, voxel_size)


# ── Point-to-Point ICP (original) ────────────────────────────────────────────

def _my_icp_point_to_point(source_down, target_down, trans_init, voxel_size):
    """
    Point-to-Point ICP using scipy KDTree + SVD.
    """
    threshold  = voxel_size * 2
    max_iters  = 20
    tol        = 1e-4

    src_pts = np.asarray(source_down.points, dtype=np.float64)
    tgt_pts = np.asarray(target_down.points, dtype=np.float64)

    T_cum = trans_init.astype(np.float64).copy()
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    src_t = (T_cum @ src_h.T).T[:, :3]

    kd_tree = KDTree(tgt_pts)
    prev_rms = np.inf

    for _ in range(max_iters):
        dist, idx = kd_tree.query(src_t, workers=-1)
        valid = dist < threshold
        if valid.sum() < 6:
            break

        src_m = src_t[valid]
        tgt_m = tgt_pts[idx[valid]]

        rms = dist[valid].mean()
        if abs(prev_rms - rms) < tol:
            break
        prev_rms = rms

        T_step = _svd_rigid_transform(src_m, tgt_m)

        T_cum = T_step @ T_cum
        src_t = (T_step[:3, :3] @ src_t.T).T + T_step[:3, 3]

    class _ICPResult:
        def __init__(self, T):
            self.transformation = T

    return _ICPResult(T_cum)


# ── Point-to-Plane ICP (new) ─────────────────────────────────────────────────

def _my_icp_point_to_plane(source_down, target_down, trans_init, voxel_size):
    """
    Point-to-Plane ICP using the linearised formulation.

    At each iteration we find closest-point correspondences, then minimise:
        sum_i  ( (R p_i + t - q_i) · n_i )^2

    Using the small-angle approximation for R (angles α, β, γ ≈ 0):
        R ≈ I + [α, β, γ]×

    This turns the problem into a linear system  A x = b  where
        x = [α, β, γ, tx, ty, tz]^T   (6 unknowns)

    For each correspondence (p, q, n):
        row of A = [ (p × n), n ]      (1×6)
        row of b = [ (q - p) · n ]     (scalar)

    We solve the 6×6 normal equations (A^T A) x = A^T b.

    Requires normals on the *target* point cloud.
    """
    threshold  = voxel_size * 2
    max_iters  = 20 #TEST
    tol        = 1e-4

    src_pts = np.asarray(source_down.points, dtype=np.float64)
    tgt_pts = np.asarray(target_down.points, dtype=np.float64)

    # Ensure target has normals
    if not target_down.has_normals():
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    tgt_normals = np.asarray(target_down.normals, dtype=np.float64)

    T_cum = trans_init.astype(np.float64).copy()
    # Apply initial transform to source points
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    src_t = (T_cum @ src_h.T).T[:, :3]

    kd_tree = KDTree(tgt_pts)
    prev_rms = np.inf

    for _ in range(max_iters):
        # Step 1 — find nearest neighbours
        dist, idx = kd_tree.query(src_t, workers=-1)
        valid = dist < threshold
        if valid.sum() < 6:
            break

        p = src_t[valid]           # (K, 3) — transformed source points
        q = tgt_pts[idx[valid]]    # (K, 3) — closest target points
        n = tgt_normals[idx[valid]]  # (K, 3) — target normals

        # Step 2 — convergence check
        # Point-to-plane residual: r_i = (p_i - q_i) · n_i
        residuals = np.sum((p - q) * n, axis=1)   # (K,)
        rms = np.sqrt(np.mean(residuals ** 2))
        if abs(prev_rms - rms) < tol:
            break
        prev_rms = rms

        # Step 3 — build linear system
        #   For each pair (p_i, q_i, n_i):
        #     a_i = [ (p_i × n_i) | n_i ]   shape (6,)
        #     b_i = (q_i - p_i) · n_i        scalar
        #
        #   Normal equations: (A^T A) x = A^T b
        cross = np.cross(p, n)          # (K, 3)
        A = np.hstack([cross, n])       # (K, 6)
        b = np.sum((q - p) * n, axis=1) # (K,)

        AtA = A.T @ A                   # (6, 6)
        Atb = A.T @ b                   # (6,)

        # Solve for x = [α, β, γ, tx, ty, tz]
        try:
            x = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            break  # singular matrix — stop

        alpha, beta, gamma, tx, ty, tz = x

        # Step 4 — construct incremental transform
        # Small-angle rotation matrix
        R_inc = np.array([
            [1,      -gamma,  beta ],
            [gamma,   1,     -alpha],
            [-beta,   alpha,  1    ],
        ])
        # Re-orthogonalise via SVD to keep R valid over many iterations
        U, _, Vt = np.linalg.svd(R_inc)
        R_inc = U @ Vt
        if np.linalg.det(R_inc) < 0:
            Vt[-1, :] *= -1
            R_inc = U @ Vt

        T_step = np.eye(4)
        T_step[:3, :3] = R_inc
        T_step[:3, 3]  = [tx, ty, tz]

        # Step 5 — accumulate and apply
        T_cum = T_step @ T_cum
        src_t = (R_inc @ src_t.T).T + np.array([tx, ty, tz])

    class _ICPResult:
        def __init__(self, T):
            self.transformation = T

    return _ICPResult(T_cum)


# ─────────────────────────────────────────────────────────────────────────────
# GT-pose helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rot(qw, qx, qy, qz):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _gt_positions_in_icp_frame(gt_poses):
    x0, y0, z0, qw0, qx0, qy0, qz0 = gt_poses[0]
    R0  = _quat_to_rot(qw0, qx0, qy0, qz0)
    t0  = np.array([x0, y0, z0])

    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3]  = t0
    T0_inv = np.linalg.inv(T0)

    positions = []
    for pose in gt_poses:
        p_world = np.append(pose[:3], 1.0)
        p_cam0_gl = (T0_inv @ p_world)[:3]
        p_cam0_cv = GL2CV @ p_cam0_gl
        positions.append(p_cam0_cv)

    return np.array(positions)


# ─────────────────────────────────────────────────────────────────────────────
# Full reconstruction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct(args):
    data_root  = args.data_root
    voxel_size = 0.05
    icp_thresh = voxel_size * 1.5

    depth_dir = os.path.join(data_root, 'depth')
    rgb_dir   = os.path.join(data_root, 'rgb')
    gt_poses  = np.load(os.path.join(data_root, 'GT_pose.npy'))
    n_frames  = len(gt_poses)

    print(f"[reconstruct] version={args.version}, icp_method={args.icp_method}, "
          f"frames={n_frames}, voxel_size={voxel_size}")

    def load_frame(idx_1based):
        rgb_path   = os.path.join(rgb_dir,   f'{idx_1based}.png')
        depth_path = os.path.join(depth_dir, f'{idx_1based}.png')

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m   = depth_raw / 1000.0
        return rgb, depth_m

    T_cum        = np.eye(4)
    pred_cam_pos = [T_cum[:3, 3].copy()]
    result_pcd   = o3d.geometry.PointCloud()

    rgb0, depth0 = load_frame(1)
    pcd_prev = depth_image_to_point_cloud(rgb0, depth0)
    prev_down, prev_fpfh = preprocess_point_cloud(pcd_prev, voxel_size)
    result_pcd += copy.deepcopy(pcd_prev)

    for i in range(1, n_frames):
        print(f"\r  frame {i}/{n_frames-1}", end='', flush=True)

        rgb_i, depth_i = load_frame(i + 1)
        pcd_curr = depth_image_to_point_cloud(rgb_i, depth_i)

        src_down, src_fpfh = preprocess_point_cloud(pcd_curr, voxel_size)
        tgt_down, tgt_fpfh = prev_down, prev_fpfh

        ransac = execute_global_registration(src_down, tgt_down,
                                             src_fpfh, tgt_fpfh,
                                             voxel_size)

        if args.version == 'open3d':
            icp_res = local_icp_algorithm(src_down, tgt_down,
                                          ransac.transformation, icp_thresh)
        else:   # 'my_icp'
            icp_res = my_local_icp_algorithm(src_down, tgt_down,
                                             ransac.transformation, voxel_size,
                                             icp_method=args.icp_method)

        T_icp = icp_res.transformation
        T_cum = T_cum @ T_icp
        pred_cam_pos.append(T_cum[:3, 3].copy())

        pcd_curr_global = copy.deepcopy(pcd_curr)
        pcd_curr_global.transform(T_cum)
        result_pcd += pcd_curr_global

        pcd_prev = pcd_curr
        prev_down, prev_fpfh = src_down, src_fpfh

    print()

    result_pcd = result_pcd.voxel_down_sample(voxel_size * 2)
    pred_cam_pos = np.array(pred_cam_pos)
    return result_pcd, pred_cam_pos


def run_with_timing(func, *args, **kwargs):
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
    parser.add_argument('--icp_method', type=str, default='point_to_plane',
                        choices=['point_to_point', 'point_to_plane'],
                        help='ICP method for custom ICP: point_to_point or point_to_plane')
    parser.add_argument('--data_root', type=str,
                        default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = 'data_collection/first_floor/'
    elif args.floor == 2:
        args.data_root = 'data_collection/second_floor/'

    (result_pcd, pred_cam_pos), elapsed_sec = run_with_timing(reconstruct, args)
    print(f"Reconstruction runtime: {elapsed_sec:.2f} s ({elapsed_sec / 60.0:.2f} min)")

    gt_poses = np.load(os.path.join(args.data_root, 'GT_pose.npy'))
    gt_pos   = _gt_positions_in_icp_frame(gt_poses)
    n        = len(pred_cam_pos)
    gt_pos   = gt_pos[:n]

    l2_per_frame = np.linalg.norm(pred_cam_pos - gt_pos, axis=1)
    mean_l2      = l2_per_frame.mean()
    print(f"Mean L2 distance: {mean_l2:.4f} m")

    ceil_above = 0.6
    pts  = np.asarray(result_pcd.points)
    cols = np.asarray(result_pcd.colors)
    mask = pts[:, 1] > -ceil_above
    trimmed = o3d.geometry.PointCloud()
    trimmed.points = o3d.utility.Vector3dVector(pts[mask])
    trimmed.colors = o3d.utility.Vector3dVector(cols[mask])

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

    est_traj = make_lineset(pred_cam_pos, [1, 0, 0])
    gt_traj  = make_lineset(gt_pos,       [0, 0, 0])

    print("Opening 3D viewer …  (close the window to exit)")
    o3d.visualization.draw_geometries(
        [trimmed, est_traj, gt_traj],
        window_name='3D Scene Reconstruction',
        width=1280, height=720,
    )