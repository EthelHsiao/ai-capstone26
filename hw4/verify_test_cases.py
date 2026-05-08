import argparse
import json
import os

import numpy as np

from fk import get_ur5_DH_params, your_fk
from ik import your_ik


BASE_POS = np.asarray([-0.2, 0.13, 0.6], dtype=np.float64)
FK_ERROR_THRESH = 0.005
JACOBIAN_ERROR_THRESH = 0.05
IK_ERROR_THRESH = 0.02


def _load_json(path):
    with open(path, "r") as f_in:
        return json.load(f_in)


def _print_result(path, errors, threshold, label):
    errors = np.asarray(errors, dtype=np.float64)
    fail_count = int(np.sum(errors > threshold))
    print(f"- {os.path.basename(path)}")
    print(f"  {label} mean : {float(np.mean(errors)):.10f}")
    print(f"  {label} max  : {float(np.max(errors)):.10f}")
    print(f"  threshold   : {threshold:.10f}")
    print(f"  fail count  : {fail_count} / {errors.size}")
    return fail_count


def verify_fk():
    dh_params = get_ur5_DH_params()
    testcase_files = [
        "test_case/fk_test_case_easy.json",
        "test_case/fk_test_case_medium.json",
        "test_case/fk_test_case_hard.json",
    ]

    total_pose_fail = 0
    total_jacobian_fail = 0

    print("==================== FK / Jacobian ====================")
    for testcase_file in testcase_files:
        fk_dict = _load_json(testcase_file)
        pose_errors = []
        jacobian_errors = []

        for q, gt_pose, gt_jacobian in zip(
            fk_dict["joint_poses"],
            fk_dict["poses"],
            fk_dict["jacobian"],
        ):
            pose, jacobian = your_fk(dh_params, q, BASE_POS)
            pose_errors.append(np.linalg.norm(np.asarray(pose) - np.asarray(gt_pose), ord=2))
            jacobian_errors.append(
                np.linalg.norm(np.asarray(jacobian) - np.asarray(gt_jacobian), ord=2)
            )

        total_pose_fail += _print_result(
            testcase_file, pose_errors, FK_ERROR_THRESH, "pose"
        )
        total_jacobian_fail += _print_result(
            testcase_file, jacobian_errors, JACOBIAN_ERROR_THRESH, "jacobian"
        )

    print("-------------------------------------------------------")
    print(f"FK pose failures      : {total_pose_fail}")
    print(f"Jacobian failures     : {total_jacobian_fail}")
    return total_pose_fail + total_jacobian_fail


def verify_ik():
    dh_params = get_ur5_DH_params()
    testcase_files = [
        "test_case/ik_test_case_easy.json",
        "test_case/ik_test_case_medium.json",
        "test_case/ik_test_case_hard.json",
    ]

    total_ik_fail = 0

    print("========================= IK ==========================")
    for testcase_file in testcase_files:
        ik_dict = _load_json(testcase_file)
        q_curr = np.asarray(ik_dict["current_joint_poses"][0], dtype=np.float64)
        ik_errors = []

        for target_pose in ik_dict["next_poses"]:
            q_curr = np.asarray(
                your_ik(target_pose, BASE_POS, q_init=q_curr),
                dtype=np.float64,
            )
            solved_pose, _ = your_fk(dh_params, q_curr, BASE_POS)
            ik_errors.append(
                np.linalg.norm(np.asarray(solved_pose) - np.asarray(target_pose), ord=2)
            )

        total_ik_fail += _print_result(
            testcase_file, ik_errors, IK_ERROR_THRESH, "ik"
        )

    print("-------------------------------------------------------")
    print(f"IK failures           : {total_ik_fail}")
    return total_ik_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", "fk", "ik"],
        default="all",
        help="which public JSON test cases to verify",
    )
    args = parser.parse_args()

    total_fail = 0
    if args.task in ("all", "fk"):
        total_fail += verify_fk()
    if args.task in ("all", "ik"):
        total_fail += verify_ik()

    print("====================== Summary =======================")
    if total_fail == 0:
        print("All checked public JSON test cases passed.")
    else:
        print(f"Total failures: {total_fail}")


if __name__ == "__main__":
    main()
