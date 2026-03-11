import argparse
import glob
import os
import cv2 as cv
from src.vision.video import downscale_frame, draw_pose_landmarks_on_image
from src.vision.process import PoseLandmarker, pose_options, create_mp_image
from src.calculation.geometric import human_angles
from src.calculation.inverse_kinematics import solve_ik
from src.robot.coppelia import FrankaPanda

VIDEOS_DIR = "./videos"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', type=int, nargs='+', choices=range(1, 8),
                        metavar='N', help='Joint(s) to activate (1-7). Default: all.')
    parser.add_argument('--video', nargs='+', metavar='FILE',
                        help='Video file(s) to analyze. Default: all *.mp4 in ./videos/')
    parser.add_argument('--inverse', action='store_true',
                        help='Use inverse kinematics mode: target wrist landmark position.')
    args = parser.parse_args()
    active = set(args.joint) if args.joint else set(range(1, 8))

    if args.video:
        video_files = [os.path.join(VIDEOS_DIR, f) for f in args.video]
    else:
        video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not video_files:
        print(f"No MP4 files found in {VIDEOS_DIR}")
        return

    robot = FrankaPanda()
    prev_ik_angles = None
    ik_frame_counter = 0

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        frame_count = 0
        for path in video_files:
            name = os.path.basename(path)
            cap = cv.VideoCapture(path)
            if not cap.isOpened():
                print(f"Could not open {path}, skipping.")
                continue

            fps = cap.get(cv.CAP_PROP_FPS) or 30
            wait_ms = max(1, int(1000 / fps))

            computer = human_angles()

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
                if args.inverse:
                    ik_frame_counter += 1
                    if ik_frame_counter % 5 == 1:
                        wrist = lm[16]   # _RIGHT_WRIST = 16
                        shoulder = lm[12]  # _RIGHT_SHOULDER = 12
                        dx = wrist.x - shoulder.x
                        dy = wrist.y - shoulder.y
                        dz = wrist.z - shoulder.z
                        # Remap: robot_x=-hand_y, robot_y=+hand_z, robot_z=-hand_x (right-hand convention)
                        joints = solve_ik((-dy, dz, -dx), initial_angles=prev_ik_angles)
                        prev_ik_angles = joints
                        robot.set_joint_angles(joints)
                    joints = prev_ik_angles or [0.0] * 7
                else:
                    computer.update_vectors(lm)
                    joints = [
                        computer.theta_1(),
                        computer.theta_2(),
                        computer.theta_3(),
                        -computer.theta_4(),
                        computer.theta_5(),
                        computer.theta_6(),
                        computer.theta_7()
                    ]
                    masked = [j if (i + 1) in active else 0.0 for i, j in enumerate(joints)]
                    robot.set_joint_angles(masked)

                draw_pose_landmarks_on_image(rgb_frame, result, joints, active, inverse=args.inverse)
                frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                cv.imshow(name, frame)
                key = cv.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()
