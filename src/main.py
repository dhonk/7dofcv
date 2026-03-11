import argparse
import glob
import math
import os
import time
import cv2 as cv
from src.vision.video import downscale_frame, draw_pose_landmarks_on_image
from src.vision.process import PoseLandmarker, pose_options, create_mp_image
from src.calculation.geometric import human_angles
from src.calculation.inverse_kinematics import solve_ik, get_home_position, get_arm_reach
from src.robot.coppelia import FrankaPanda
from src.data_processing.save_data import save_arm_csv, plot_all, plot_compare, make_run_dir

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
    args = parser.parse_args()
    active = set(args.joint) if args.joint else set(range(1, 8))

    if args.video:
        video_files = [os.path.join(VIDEOS_DIR, f) for f in args.video]
    else:
        video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not video_files:
        print(f"No MP4 files found in {VIDEOS_DIR}")
        return

    ik_home     = get_home_position()
    robot_reach = get_arm_reach()

    def _dist3d(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    if args.compare:
        robot = FrankaPanda()
        fk_records, ik_records = [], []
        fk_compute_time = ik_compute_time = 0.0
        cached_lms = []

        # --- FK Pass ---
        with PoseLandmarker.create_from_options(pose_options) as landmarker:
            frame_count = 0
            for path in video_files:
                name = os.path.basename(path)
                cap = cv.VideoCapture(path)
                if not cap.isOpened():
                    print(f"Could not open {path}, skipping.")
                    continue
                fps = cap.get(cv.CAP_PROP_FPS) or 30
                computer = human_angles()
                fk_masked = [0.0] * 7
                start_time = time.perf_counter()
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

                    if frame_count % 5 == 1:
                        t0 = time.perf_counter()
                        computer.update_vectors(lm)
                        fk_thetas = [
                            computer.theta_1(), computer.theta_2(), computer.theta_3(),
                            computer.theta_4(), computer.theta_5(), computer.theta_6(),
                            computer.theta_7()
                        ]
                        fk_joints = list(fk_thetas)
                        fk_joints[3] = -fk_joints[3]
                        fk_masked = [j if (i + 1) in active else 0.0 for i, j in enumerate(fk_joints)]
                        fk_compute_time += time.perf_counter() - t0
                        robot.set_joint_angles(fk_masked)

                    positions = robot.get_joint_positions_3d()
                    fk_records.append({
                        "frame": frame_count - 1,
                        **{f"joint{j+1}_x": positions[j][0] for j in range(7)},
                        **{f"joint{j+1}_y": positions[j][1] for j in range(7)},
                        **{f"joint{j+1}_z": positions[j][2] for j in range(7)},
                    })

                    draw_pose_landmarks_on_image(rgb_frame, result, fk_masked, active, inverse=False)
                    frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                    if not args.headless:
                        cv.imshow(f"FK - {name}", frame)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break
                elapsed = time.perf_counter() - start_time
                print(f"[FK] [{name}] processed {frame_count} frames in {elapsed:.2f}s")
                cap.release()
                cv.destroyAllWindows()

        # --- IK Pass ---
        prev_ik_angles = None
        start_time = time.perf_counter()
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
            ik_records.append({
                "frame": i,
                **{f"joint{j+1}_x": positions[j][0] for j in range(7)},
                **{f"joint{j+1}_y": positions[j][1] for j in range(7)},
                **{f"joint{j+1}_z": positions[j][2] for j in range(7)},
            })
        elapsed = time.perf_counter() - start_time
        print(f"[IK] processed {len(cached_lms)} frames in {elapsed:.2f}s")

        savings = abs(fk_compute_time - ik_compute_time)
        faster = "FK" if fk_compute_time < ik_compute_time else "IK"
        print(f"FK total compute time: {fk_compute_time:.3f}s")
        print(f"IK total compute time: {ik_compute_time:.3f}s")
        print(f"{faster} was faster by {savings:.3f}s ({savings / max(fk_compute_time, ik_compute_time) * 100:.1f}%)")
        plot_compare(fk_records, ik_records, headless=args.headless)
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
                print(f"Could not open {path}, skipping.")
                continue

            fps = cap.get(cv.CAP_PROP_FPS) or 30
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
                        robot.set_joint_angles(masked)

                if args.inverse:
                    joints = prev_ik_angles or [0.0] * 7
                    masked = joints

                robot_angles = robot.get_joint_angles()
                robot_forces = robot.get_joint_forces()
                robot_vels = robot.get_joint_vel()

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
                cv.imshow(name, frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            elapsed = time.perf_counter() - start_time
            print(f"[{name}] processed {frame_count} frames in {elapsed:.2f}s")
            cap.release()
            if writer is not None:
                writer.release()
            cv.destroyAllWindows()

    save_arm_csv(records, tag=run_tag)
    plot_all(records, active, save_dir=save_dir)

if __name__ == "__main__":
    main()
