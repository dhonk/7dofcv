import argparse
import glob
import os
import cv2 as cv
from src.vision.video import downscale_frame, draw_pose_landmarks_on_image
from src.vision.process import PoseLandmarker, pose_options, create_mp_image
from src.calculation.geometric import human_angles
from src.robot.coppelia import FrankaPanda

VIDEOS_DIR = "./videos"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', type=int, nargs='+', choices=range(1, 8),
                        metavar='N', help='Joint(s) to activate (1-7). Default: all.')
    args = parser.parse_args()
    active = set(args.joint) if args.joint else set(range(1, 8))
    video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not video_files:
        print(f"No MP4 files found in {VIDEOS_DIR}")
        return

    robot = FrankaPanda()

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

                draw_pose_landmarks_on_image(rgb_frame, result, joints, active)
                frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                cv.imshow(name, frame)
                key = cv.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()

if __name__ == "__main__":
    main()
