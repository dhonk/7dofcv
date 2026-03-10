import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


ArmRecord = dict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CSV_FIELDS = [
    "frame",
    "shoulder_x", "shoulder_y", "shoulder_z",
    "elbow_x",    "elbow_y",    "elbow_z",
    "wrist_x",    "wrist_y",    "wrist_z",
    "index_x",    "index_y",    "index_z",
    "hand_wrist_x",     "hand_wrist_y",     "hand_wrist_z",
    "hand_index_mcp_x", "hand_index_mcp_y", "hand_index_mcp_z",
    "hand_ring_mcp_x",  "hand_ring_mcp_y",  "hand_ring_mcp_z",
    "human_shoulder_hori_angle",
    "robot_joint1_angle",
    "human_shoulder_vert_angle",
    "robot_joint2_angle",
    "human_elbow_angle",
    "robot_joint4_angle",
    "human_wrist_flex_angle",
    "robot_joint6_angle",
    "human_wrist_rot_angle",
    "robot_joint5_angle",
    "sent_joint1", "sent_joint2", "sent_joint3", "sent_joint4",
    "sent_joint5", "sent_joint6", "sent_joint7",
]


def save_arm_csv(records: list[ArmRecord], path: str | None = None) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(
            RESULTS_DIR,
            f"arm_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} frames to {path}")


def plot_arm_tracking(records: list[ArmRecord],
                      active_joints: set[int] | None = None) -> None:
    if not records:
        return
    if active_joints is None:
        active_joints = set(range(1, 8))

    frames = [r["frame"] for r in records]

    # Pose world landmark positions
    pose_parts = {
        "Right Shoulder":     ("shoulder_x", "shoulder_y", "shoulder_z"),
        "Right Elbow":       ("elbow_x",    "elbow_y",    "elbow_z"),
        "Right Wrist":       ("wrist_x",    "wrist_y",    "wrist_z"),
        "Right Index Finger": ("index_x",    "index_y",    "index_z"),
    }
    fig1, axes1 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig1.suptitle("Pose Landmark Positions")
    for ax, (part, (xk, yk, zk)) in zip(axes1, pose_parts.items()):
        ax.plot(frames, [r[xk] for r in records], label="x")
        ax.plot(frames, [r[yk] for r in records], label="y")
        ax.plot(frames, [r[zk] for r in records], label="z")
        ax.set_title(part)
        ax.set_ylabel("Position (normalized)")
        ax.legend()
        ax.grid(True)
    axes1[-1].set_xlabel("Frame")
    fig1.tight_layout()

    # Hand landmark positions
    hand_parts = {
        "Hand Wrist (0)":     ("hand_wrist_x",     "hand_wrist_y",     "hand_wrist_z"),
        "Index Finger MCP (5)": ("hand_index_mcp_x", "hand_index_mcp_y", "hand_index_mcp_z"),
        "Ring Finger MCP (13)": ("hand_ring_mcp_x",  "hand_ring_mcp_y",  "hand_ring_mcp_z"),
    }
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig3.suptitle("Hand Landmark Positions")
    for ax, (part, (xk, yk, zk)) in zip(axes3, hand_parts.items()):
        ax.plot(frames, [r.get(xk, 0) for r in records], label="x")
        ax.plot(frames, [r.get(yk, 0) for r in records], label="y")
        ax.plot(frames, [r.get(zk, 0) for r in records], label="z")
        ax.set_title(part)
        ax.set_ylabel("Position (normalized)")
        ax.legend()
        ax.grid(True)
    axes3[-1].set_xlabel("Frame")
    fig3.tight_layout()

    # Sent joint angles for active joints
    joint_keys = [f"sent_joint{j}" for j in sorted(active_joints)]
    fig2, axes2 = plt.subplots(len(joint_keys), 1, figsize=(12, 3 * len(joint_keys)),
                               sharex=True, squeeze=False)
    fig2.suptitle("Sent Joint Angles (Active Joints)")
    for ax, key in zip(axes2[:, 0], joint_keys):
        ax.plot(frames, [r.get(key, 0) for r in records])
        ax.set_title(key.replace("_", " ").title())
        ax.set_ylabel("Angle (rad)")
        ax.grid(True)
    axes2[-1, 0].set_xlabel("Frame")
    fig2.tight_layout()

    plt.show()
