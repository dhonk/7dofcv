import argparse
import glob
import logging
import math
import os
import time
import cv2 as cv
from src.vision.video import downscale_frame, draw_pose_landmarks_on_image
from src.vision.process import PoseLandmarker, pose_options, create_mp_image
from src.calculation.geometric import human_angles
from src.calculation.inverse_kinematics import solve_ik, get_home_position, get_arm_reach
from src.robot.coppelia import FrankaPanda
from src.data_processing.save_data import save_arm_csv, plot_all, plot_compare, plot_compare_eef, make_run_dir

VIDEOS_DIR = "./videos"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', type=int, nargs='+', choices=range(1, 8),
                        metavar='N', help='Joint(s) to activate (1-7). Default: all.')
    parser.add_argument('--video', nargs='+', metavar='FILE',
                        help='Video file(s) to analyze. Default: all *.mp4 in ./videos/')
    parser.add_argument('--inverse', action='store_true',
                        help='Use inverse kinematics mode: target wrist landmark position.')
    parser.add_argument('--compare', action='store_true',
                        help='Compare IK vs geometric approach: timing + joint angle graph.')
    parser.add_argument('--headless', action='store_true',
                        help='Suppress cv.imshow and plt.show for batch/non-display runs.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable detailed logging')
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.WARNING,
    )
    if args.verbose:
        for name in ("src", "__main__"):
            logging.getLogger(name).setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)

    active = set(args.joint) if args.joint else set(range(1, 8))
    logger.debug("Active joints: %s", sorted(active))
    logger.debug("Mode: %s", "compare" if args.compare else "IK" if args.inverse else "FK")

    if args.video:
        video_files = [os.path.join(VIDEOS_DIR, f) for f in args.video]
    else:
        video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not video_files:
        logger.warning("No MP4 files found in %s", VIDEOS_DIR)
        return
    logger.debug("Video files: %s", video_files)

    ik_home     = get_home_position()
    robot_reach = get_arm_reach()

    def _dist3d(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    if args.compare:
        robot = FrankaPanda()
        fk_records, ik_records = [], []
        fk_compute_time = ik_compute_time = 0.0
        cached_lms = []

        # --- Video decode + MediaPipe (shared overhead) ---
        with PoseLandmarker.create_from_options(pose_options) as landmarker:
            frame_count = 0
            for path in video_files:
                name = os.path.basename(path)
                cap = cv.VideoCapture(path)
                if not cap.isOpened():
                    logger.warning("Could not open %s, skipping.", path)
                    continue
                fps = cap.get(cv.CAP_PROP_FPS) or 30
                logger.debug("Opened %s (fps=%.1f)", name, fps)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = downscale_frame(frame)
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    mp_image = create_mp_image(rgb_frame)
                    timestamp_ms = int(frame_count / fps * 1000)
                    frame_count += 1
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    lm = result.pose_landmarks[0]
                    cached_lms.append(lm)
                cap.release()
            logger.info("Decoded %d frames from %d video(s)", frame_count, len(video_files))

        # --- FK Pass (uses cached landmarks) ---
        computer = human_angles()
        fk_masked = [0.0] * 7
        fk_start = time.perf_counter()
        for i, lm in enumerate(cached_lms):
            if i % 5 == 0:
                t0 = time.perf_counter()
                computer.update_vectors(lm)
                fk_thetas = [
                    computer.theta_1(), computer.theta_2(), computer.theta_3(),
                    computer.theta_4(), computer.theta_5(), computer.theta_6(),
                    computer.theta_7()
                ]
                fk_joints = list(fk_thetas)
                fk_joints[3] = -fk_joints[3]
                fk_masked = [j if (idx + 1) in active else 0.0 for idx, j in enumerate(fk_joints)]
                fk_compute_time += time.perf_counter() - t0
                robot.set_joint_angles(fk_masked)

            positions = robot.get_joint_positions_3d()
            eef_pos, eef_ori = robot.get_eef_pose()
            fk_records.append({
                "frame": i,
                "time": time.perf_counter() - fk_start,
                **{f"joint{j+1}_x": positions[j][0] for j in range(7)},
                **{f"joint{j+1}_y": positions[j][1] for j in range(7)},
                **{f"joint{j+1}_z": positions[j][2] for j in range(7)},
                "eef_x": eef_pos[0], "eef_y": eef_pos[1], "eef_z": eef_pos[2],
                "eef_alpha": eef_ori[0], "eef_beta": eef_ori[1], "eef_gamma": eef_ori[2],
            })
        fk_wall_time = time.perf_counter() - fk_start

        # --- IK Pass (uses cached landmarks) ---
        prev_ik_angles = None
        ik_start = time.perf_counter()
        for i, lm in enumerate(cached_lms):
            sh = lm[12]; el = lm[14]; wr = lm[16]
            if i % 5 == 0:
                t0 = time.perf_counter()
                arm_len = max(_dist3d(sh, el) + _dist3d(el, wr), 1e-3)
                dx = (wr.x - sh.x) / arm_len
                dy = (wr.y - sh.y) / arm_len
                dz = (wr.z - sh.z) / arm_len
                target = (
                    ik_home[0] + (-dz) * robot_reach,
                    ik_home[1] + (-dx) * robot_reach,
                    ik_home[2] + (-dy) * robot_reach,
                )
                ik_joints = solve_ik(target, initial_angles=prev_ik_angles)
                prev_ik_angles = ik_joints
                robot.set_joint_angles(ik_joints)
                ik_compute_time += time.perf_counter() - t0

            positions = robot.get_joint_positions_3d()
            eef_pos, eef_ori = robot.get_eef_pose()
            ik_records.append({
                "frame": i,
                "time": time.perf_counter() - ik_start,
                **{f"joint{j+1}_x": positions[j][0] for j in range(7)},
                **{f"joint{j+1}_y": positions[j][1] for j in range(7)},
                **{f"joint{j+1}_z": positions[j][2] for j in range(7)},
                "eef_x": eef_pos[0], "eef_y": eef_pos[1], "eef_z": eef_pos[2],
                "eef_alpha": eef_ori[0], "eef_beta": eef_ori[1], "eef_gamma": eef_ori[2],
            })
        ik_wall_time = time.perf_counter() - ik_start

        n = len(cached_lms)
        logger.info("[FK] Processed %d frames — wall: %.2fs, compute: %.3fs", n, fk_wall_time, fk_compute_time)
        logger.info("[IK] Processed %d frames — wall: %.2fs, compute: %.3fs", n, ik_wall_time, ik_compute_time)
        savings = abs(fk_compute_time - ik_compute_time)
        faster = "FK" if fk_compute_time < ik_compute_time else "IK"
        pct = savings / max(fk_compute_time, ik_compute_time) * 100 if max(fk_compute_time, ik_compute_time) > 0 else 0
        logger.info("Compute comparison: FK %.3fs vs IK %.3fs — %s faster by %.3fs (%.1f%%)", fk_compute_time, ik_compute_time, faster, savings, pct)

        save_dir = make_run_dir(tag="compare")

        summary_path = os.path.join(save_dir, "compare_results.txt")
        with open(summary_path, "w") as f:
            f.write(f"FK vs IK Comparison\n")
            f.write(f"Frames: {n}\n\n")
            f.write(f"[FK] Wall: {fk_wall_time:.2f}s, Compute: {fk_compute_time:.3f}s\n")
            f.write(f"[IK] Wall: {ik_wall_time:.2f}s, Compute: {ik_compute_time:.3f}s\n\n")
            f.write(f"{faster} faster by {savings:.3f}s ({pct:.1f}%)\n")
        logger.info("Compare results saved to %s", summary_path)

        plot_compare(fk_records, ik_records, save_dir=save_dir, headless=args.headless)
        plot_compare_eef(fk_records, ik_records, save_dir=save_dir, headless=args.headless)
        return

    robot = FrankaPanda()
    prev_ik_angles = None
    ik_frame_counter = 0
    records = []

    video_stems = [os.path.splitext(os.path.basename(p))[0] for p in video_files]
    videos_tag = "+".join(video_stems) if len(video_stems) <= 3 else f"{video_stems[0]}+{len(video_stems)-1}more"
    if args.inverse:
        joints_tag = "ik"
    elif active == set(range(1, 8)):
        joints_tag = "all"
    else:
        joints_tag = "j" + "".join(str(j) for j in sorted(active))
    run_tag = f"{videos_tag}_{joints_tag}"

    save_dir = make_run_dir(tag=run_tag)

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        frame_count = 0
        for path in video_files:
            name = os.path.basename(path)
            cap = cv.VideoCapture(path)
            if not cap.isOpened():
                logger.warning("Could not open %s, skipping.", path)
                continue

            fps = cap.get(cv.CAP_PROP_FPS) or 30
            logger.debug("Opened %s (fps=%.1f)", name, fps)
            out_path = os.path.join(save_dir, f"landmarks_{os.path.splitext(name)[0]}.mp4")
            writer = None

            computer = human_angles()
            masked = [0.0] * 7
            joints = [0.0] * 7
            human_thetas = [0.0] * 7

            start_time = time.perf_counter()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = downscale_frame(frame)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = create_mp_image(rgb_frame)
                timestamp_ms = int(frame_count / fps * 1000)
                frame_count += 1
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                lm = result.pose_landmarks[0]
                ls = lm[11]   # LEFT_SHOULDER
                sh = lm[12]   # RIGHT_SHOULDER
                el = lm[14]   # RIGHT_ELBOW
                wr = lm[16]   # RIGHT_WRIST
                if frame_count % 5 == 1:
                    if args.inverse:
                        ik_frame_counter += 1
                        arm_len = max(_dist3d(sh, el) + _dist3d(el, wr), 1e-3)
                        dx = (wr.x - sh.x) / arm_len   # lateral
                        dy = (wr.y - sh.y) / arm_len   # vertical
                        dz = (wr.z - sh.z) / arm_len   # depth

                        target = (
                            ik_home[0] + (-dz) * robot_reach,   # x: forward
                            ik_home[1] + (-dx) * robot_reach,   # y: lateral
                            ik_home[2] + (-dy) * robot_reach,   # z: vertical
                        )
                        joints = solve_ik(target, initial_angles=prev_ik_angles)
                        prev_ik_angles = joints
                        logger.debug("Frame %d IK target=(%.3f,%.3f,%.3f) joints=%s",
                                     frame_count - 1, *target, [f"{j:.3f}" for j in joints])
                        robot.set_joint_angles(joints)
                    else:
                        computer.update_vectors(lm)
                        human_thetas = [
                            computer.theta_1(),
                            computer.theta_2(),
                            computer.theta_3(),
                            computer.theta_4(),
                            computer.theta_5(),
                            computer.theta_6(),
                            computer.theta_7()
                        ]
                        joints = list(human_thetas)
                        joints[3] = -joints[3]
                        masked = [j if (i + 1) in active else 0.0 for i, j in enumerate(joints)]
                        logger.debug("Frame %d FK angles=%s sent=%s",
                                     frame_count - 1, [f"{j:.3f}" for j in joints], [f"{m:.3f}" for m in masked])
                        robot.set_joint_angles(masked)

                if args.inverse:
                    joints = prev_ik_angles or [0.0] * 7
                    masked = joints

                robot_angles = robot.get_joint_angles()
                robot_forces = robot.get_joint_forces()
                robot_vels = robot.get_joint_vel()
                logger.debug("Frame %d robot feedback angles=%s",
                             frame_count - 1, [f"{a:.3f}" for a in robot_angles])

                # Build landmark coords from pose result
                el, ix = lm[14], lm[20]
                record = {
                    "frame": frame_count - 1,
                    "left_shoulder_x": ls.x, "left_shoulder_y": ls.y, "left_shoulder_z": ls.z,
                    "shoulder_x": sh.x, "shoulder_y": sh.y, "shoulder_z": sh.z,
                    "elbow_x":    el.x, "elbow_y":    el.y, "elbow_z":    el.z,
                    "wrist_x":    wr.x, "wrist_y":    wr.y, "wrist_z":    wr.z,
                    "index_x":    ix.x, "index_y":    ix.y, "index_z":    ix.z,
                    **{f"lm{i}_x": pt.x for i, pt in enumerate(lm)},
                    **{f"lm{i}_y": pt.y for i, pt in enumerate(lm)},
                    **{f"lm{i}_z": pt.z for i, pt in enumerate(lm)},
                    "human_shoulder_hori_angle": human_thetas[0],
                    "human_shoulder_vert_angle": human_thetas[1],
                    "human_elbow_angle":         human_thetas[3],
                    "human_wrist_flex_angle":    human_thetas[5],
                    "human_wrist_rot_angle":     human_thetas[4],
                    "robot_joint1_angle": robot_angles[0],
                    "robot_joint2_angle": robot_angles[1],
                    "robot_joint4_angle": robot_angles[3],
                    "robot_joint5_angle": robot_angles[4],
                    "robot_joint6_angle": robot_angles[5],
                    "sent_joint1": masked[0], "sent_joint2": masked[1], "sent_joint3": masked[2],
                    "sent_joint4": masked[3], "sent_joint5": masked[4], "sent_joint6": masked[5],
                    "sent_joint7": masked[6],
                    "robot_joint1_force": robot_forces[0], "robot_joint2_force": robot_forces[1],
                    "robot_joint3_force": robot_forces[2], "robot_joint4_force": robot_forces[3],
                    "robot_joint5_force": robot_forces[4], "robot_joint6_force": robot_forces[5],
                    "robot_joint7_force": robot_forces[6],
                    "robot_joint1_vel": robot_vels[0], "robot_joint2_vel": robot_vels[1],
                    "robot_joint3_vel": robot_vels[2], "robot_joint4_vel": robot_vels[3],
                    "robot_joint5_vel": robot_vels[4], "robot_joint6_vel": robot_vels[5],
                    "robot_joint7_vel": robot_vels[6],
                }
                records.append(record)

                draw_pose_landmarks_on_image(rgb_frame, result, joints, active, inverse=args.inverse)
                frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                writer.write(frame)
                if not args.headless:
                    cv.imshow(name, frame)
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            elapsed = time.perf_counter() - start_time
            logger.info("[%s] processed %d frames in %.2fs (%.1f fps)", name, frame_count, elapsed, frame_count / elapsed if elapsed > 0 else 0)
            cap.release()
            if writer is not None:
                writer.release()
            cv.destroyAllWindows()

    save_arm_csv(records, tag=run_tag)
    plot_all(records, active, save_dir=save_dir, headless=args.headless)

if __name__ == "__main__":
    main()
