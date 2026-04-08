import os
import re
import glob
import numpy as np
import open3d as o3d
import argparse
import cv2
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1000.0

# Habitat uses OpenGL convention (Y-up, Z-backward).
# Our depth un-projection works in OpenCV convention (Y-down, Z-forward).
# GL2CV flips Y and Z to switch between the two.
GL2CV = np.diag([1.0, -1.0, -1.0])


def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).

    The implementation follows the pinhole camera model (OpenCV convention):
        Z  = depth[v, u]             (metres)
        X  = (u - CX) * Z / FX
        Y  = (v - CY) * Z / FY
    """
    H, W = depth_image.shape

    u_grid = np.arange(W, dtype=np.float64)
    v_grid = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid)

    Z = depth_image.astype(np.float64)

    X = (uu - CX) * Z / FX
    Y = (vv - CY) * Z / FY

    valid = Z > 0.01

    points_3d = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    colors_norm = rgb_image[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Compute FPFH features for Global Registration [cite: 30]
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh, voxel_size):
    """
    Global registration with RANSAC [cite: 30]
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
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40_000, 200),
    )
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) [cite: 32]
    """
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )
    return result


# ---------- Custom ICP helpers ----------

def _svd_rigid_transform(src, tgt):
    """
    Compute the best-fit rigid transform R, t that maps src -> tgt
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

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_cum
    return result


def _my_icp_point_to_plane(source_down, target_down, trans_init, voxel_size):
    """
    Point-to-Plane ICP using the linearised formulation.

    At each iteration we find closest-point correspondences, then minimise:
        sum_i  ( (R p_i + t - q_i) . n_i )^2

    Using the small-angle approximation for R (angles a, b, g ~ 0):
        R ~ I + [a, b, g]x

    This turns the problem into a linear system  A x = b  where
        x = [a, b, g, tx, ty, tz]^T   (6 unknowns)

    Requires normals on the *target* point cloud.
    """
    threshold  = voxel_size * 2
    max_iters  = 20
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
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    src_t = (T_cum @ src_h.T).T[:, :3]

    kd_tree = KDTree(tgt_pts)
    prev_rms = np.inf

    for _ in range(max_iters):
        # Step 1 - find nearest neighbours
        dist, idx = kd_tree.query(src_t, workers=-1)
        valid = dist < threshold
        if valid.sum() < 6:
            break

        p = src_t[valid]
        q = tgt_pts[idx[valid]]
        n = tgt_normals[idx[valid]]

        # Step 2 - convergence check
        residuals = np.sum((p - q) * n, axis=1)
        rms = np.sqrt(np.mean(residuals ** 2))
        if abs(prev_rms - rms) < tol:
            break
        prev_rms = rms

        # Step 3 - build linear system
        cross = np.cross(p, n)
        A = np.hstack([cross, n])
        b = np.sum((q - p) * n, axis=1)

        AtA = A.T @ A
        Atb = A.T @ b

        try:
            x = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            break

        alpha, beta, gamma, tx, ty, tz = x

        # Step 4 - construct incremental transform
        R_inc = np.array([
            [1,      -gamma,  beta ],
            [gamma,   1,     -alpha],
            [-beta,   alpha,  1    ],
        ])
        U, _, Vt = np.linalg.svd(R_inc)
        R_inc = U @ Vt
        if np.linalg.det(R_inc) < 0:
            Vt[-1, :] *= -1
            R_inc = U @ Vt

        T_step = np.eye(4)
        T_step[:3, :3] = R_inc
        T_step[:3, 3]  = [tx, ty, tz]

        # Step 5 - accumulate and apply
        T_cum = T_step @ T_cum
        src_t = (R_inc @ src_t.T).T + np.array([tx, ty, tz])

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_cum
    return result


def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform, voxel_size,
                           icp_method='point_to_plane'):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%)
    Implement your own version of Point-to-Plane ICP.
    """
    if icp_method == 'point_to_point':
        return _my_icp_point_to_point(source_pcd, target_pcd, initial_transform, voxel_size)
    else:
        return _my_icp_point_to_plane(source_pcd, target_pcd, initial_transform, voxel_size)


# ---------- GT-pose helpers ----------

def _quat_to_rot(qw, qx, qy, qz):
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


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


def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization [cite: 19, 35-38]
    """
    # Convert GT poses to ICP frame positions
    gt_raw = np.load(os.path.join(args.data_root, 'GT_pose.npy'))
    gt_pos = _gt_positions_in_icp_frame(gt_raw)
    n = len(predicted_cam_poses)
    gt_pos = gt_pos[:n]

    # Calculate Mean L2 Distance [cite: 38]
    l2_per_frame = np.linalg.norm(predicted_cam_poses - gt_pos, axis=1)
    mean_l2_error = l2_per_frame.mean()

    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    # Post-processing: remove the ceiling [cite: 37]
    ceil_above = 0.6
    pts  = np.asarray(reconstructed_pcd.points)
    cols = np.asarray(reconstructed_pcd.colors)
    mask = pts[:, 1] > -ceil_above
    trimmed = o3d.geometry.PointCloud()
    trimmed.points = o3d.utility.Vector3dVector(pts[mask])
    trimmed.colors = o3d.utility.Vector3dVector(cols[mask])

    # Create LineSet for estimated trajectory (Red)
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

    est_traj = make_lineset(predicted_cam_poses, [1, 0, 0])
    # Create LineSet for ground truth trajectory (Black)
    gt_traj  = make_lineset(gt_pos, [0, 0, 0])

    # Visualization
    print("Opening 3D viewer ...  (close the window to exit)")
    o3d.visualization.draw_geometries(
        [trimmed, est_traj, gt_traj],
        window_name=f"Floor {args.floor} Reconstruction",
        width=1280, height=720,
    )
    return mean_l2_error


def reconstruct(args):
    voxel_size = 0.05
    icp_thresh = voxel_size * 1.5
    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    # Load Ground Truth Poses [cite: 24, 54]
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    n_frames = len(gt_poses)
    print(f"[reconstruct] version={args.version}, frames={n_frames}, voxel_size={voxel_size}")

    camera_poses = [np.eye(4)]
    accumulated_pcd = o3d.geometry.PointCloud()
    predicted_cam_poses = [np.eye(4)[:3, 3].copy()]

    def load_frame(idx_1based):
        rgb_path = os.path.join(rgb_dir, f"{idx_1based}.png")
        depth_path = os.path.join(depth_dir, f"{idx_1based}.png")

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
        return rgb, depth

    # Load first frame
    rgb0, depth0 = load_frame(1)
    pcd_prev = depth_image_to_point_cloud(rgb0, depth0)
    prev_down, prev_fpfh = preprocess_point_cloud(pcd_prev, voxel_size)
    accumulated_pcd += deepcopy(prev_down)

    # Reconstruction Loop [cite: 29-30]
    for i in range(1, n_frames):
        print(f"\r  Processing Frame {i}/{n_frames-1}...", end='', flush=True)

        # 1. Convert RGB-D to PointCloud (Task 1)
        rgb_i, depth_i = load_frame(i + 1)
        pcd_curr = depth_image_to_point_cloud(rgb_i, depth_i)

        # 2. Preprocess (Voxel/FPFH/Normals)
        src_down, src_fpfh = preprocess_point_cloud(pcd_curr, voxel_size)
        tgt_down, tgt_fpfh = prev_down, prev_fpfh

        # 3. Execute Global Registration (RANSAC)
        ransac = execute_global_registration(src_down, tgt_down,
                                             src_fpfh, tgt_fpfh,
                                             voxel_size)

        # 4. Execute Local Registration (ICP - Task 2)
        if args.version == 'open3d':
            icp_res = local_icp_algorithm(src_down, tgt_down,
                                          ransac.transformation, icp_thresh)
        else:   # 'my_icp'
            icp_res = my_local_icp_algorithm(src_down, tgt_down,
                                             ransac.transformation, voxel_size,
                                             icp_method=args.icp_method)

        # 5. Update camera_poses and accumulate points
        T_icp = icp_res.transformation
        T_cum = camera_poses[-1] @ T_icp
        camera_poses.append(T_cum)
        predicted_cam_poses.append(T_cum[:3, 3].copy())

        src_down_global = deepcopy(src_down)
        src_down_global.transform(T_cum)
        accumulated_pcd += src_down_global

        prev_down, prev_fpfh = src_down, src_fpfh

    print()

    # Post-processing: voxel down-sample the accumulated cloud
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size * 2)
    predicted_cam_poses = np.array(predicted_cam_poses)

    return accumulated_pcd, predicted_cam_poses, gt_poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d',
                        help='open3d or my_icp')
    parser.add_argument('--icp_method', type=str, default='point_to_plane',
                        choices=['point_to_point', 'point_to_plane'],
                        help='ICP method for custom ICP: point_to_point or point_to_plane')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)

    print(f"Total execution time: {time.time() - start_time:.2f}s")
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
