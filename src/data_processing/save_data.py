import csv
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ArmRecord = dict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def make_run_dir(tag: str | None = None) -> str:
    name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        name = f"{name}_{tag}"
    path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def _make_run_dir(tag: str | None = None) -> str:
    return make_run_dir(tag)

CSV_FIELDS = [
    "frame",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
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
    "robot_joint1_force", "robot_joint2_force", "robot_joint3_force", "robot_joint4_force",
    "robot_joint5_force", "robot_joint6_force", "robot_joint7_force",
    "robot_joint1_vel", "robot_joint2_vel", "robot_joint3_vel", "robot_joint4_vel",
    "robot_joint5_vel", "robot_joint6_vel", "robot_joint7_vel",
]


def save_arm_csv(records: list[ArmRecord], path: str | None = None, tag: str | None = None) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if path is None:
        stem = f"arm_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if tag:
            stem = f"{stem}_{tag}"
        path = os.path.join(RESULTS_DIR, f"{stem}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved %d frames to %s", len(records), path)


def plot_arm_tracking(records: list[ArmRecord],
                      active_joints: set[int] | None = None,
                      save_dir: str | None = None) -> None:
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
    if save_dir:
        fig1.savefig(os.path.join(save_dir, "pose_landmarks.png"), dpi=150, bbox_inches="tight")

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
    if save_dir:
        fig3.savefig(os.path.join(save_dir, "hand_landmarks.png"), dpi=150, bbox_inches="tight")

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
    if save_dir:
        fig2.savefig(os.path.join(save_dir, "sent_joint_angles.png"), dpi=150, bbox_inches="tight")


def plot_human_angles(records: list[ArmRecord], save_dir: str | None = None) -> None:
    if not records:
        return
    frames = [r["frame"] for r in records]
    specs = [
        ("human_shoulder_hori_angle", "Shoulder Horizontal"),
        ("human_shoulder_vert_angle", "Shoulder Vertical"),
        ("human_elbow_angle",         "Elbow"),
        ("human_wrist_flex_angle",    "Wrist Flexion"),
        ("human_wrist_rot_angle",     "Wrist Rotation"),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(12, 3 * len(specs)), sharex=True)
    fig.suptitle("Human Joint Angles")
    for ax, (key, title) in zip(axes, specs):
        ax.plot(frames, [r.get(key, 0) for r in records])
        ax.set_title(title)
        ax.set_ylabel("Angle (rad)")
        ax.grid(True)
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "human_angles.png"), dpi=150, bbox_inches="tight")


def plot_robot_state(records: list[ArmRecord], save_dir: str | None = None) -> None:
    if not records:
        return
    frames = [r["frame"] for r in records]

    for label, prefix, filename in [
        ("Robot Feedback Angles",  "robot_joint{}_angle", "robot_angles.png"),
        ("Robot Joint Forces",     "robot_joint{}_force", "robot_forces.png"),
        ("Robot Joint Velocities", "robot_joint{}_vel",   "robot_velocities.png"),
    ]:
        keys = [prefix.format(j) for j in range(1, 8)]
        fig, axes = plt.subplots(7, 1, figsize=(12, 21), sharex=True)
        fig.suptitle(label)
        for ax, key in zip(axes, keys):
            ax.plot(frames, [r.get(key, 0) for r in records])
            ax.set_title(key.replace("_", " ").title())
            ax.set_ylabel("Value")
            ax.grid(True)
        axes[-1].set_xlabel("Frame")
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")


_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),   # face
    (9,10),                                              # mouth
    (11,12),                                             # shoulders
    (11,13),(13,15),(15,17),(17,19),(19,15),(15,21),    # left arm
    (12,14),(14,16),(16,18),(18,20),(20,16),(16,22),    # right arm
    (11,23),(12,24),(23,24),                             # torso
    (23,25),(24,26),(25,27),(26,28),                    # upper legs
    (27,29),(29,31),(27,31),(28,30),(30,32),(28,32),    # lower legs/feet
]

_ARM_CHAIN = [11, 12, 14, 16, 20]


def plot_pose_3d_animated(records: list[ArmRecord], save_dir: str | None = None) -> None:
    if not records:
        return

    # Compute axis limits from all 33 landmarks across all frames
    all_xs, all_ys, all_zs = [], [], []
    for r in records:
        for i in range(33):
            all_xs.append(r.get(f"lm{i}_x", 0.0))
            all_ys.append(r.get(f"lm{i}_y", 0.0))
            all_zs.append(r.get(f"lm{i}_z", 0.0))

    xlim = (min(all_xs), max(all_xs))
    ylim = (-max(all_ys), -min(all_ys))
    zlim = (min(all_zs), max(all_zs))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_zlim(ylim)
    title = ax.set_title("Frame 0")

    # Create one Line3D per connection
    conn_lines = []
    for (a, b) in _POSE_CONNECTIONS:
        ln, = ax.plot([], [], [], color='grey', lw=0.8, alpha=0.5)
        conn_lines.append((ln, a, b))

    arm_line, = ax.plot([], [], [], 'b-o', lw=2)
    scat = ax.scatter([], [], [], s=10, c='red')

    def update(i):
        r = records[i]
        xs = [r.get(f"lm{j}_x", 0.0) for j in range(33)]
        ys = [-r.get(f"lm{j}_y", 0.0) for j in range(33)]
        zs = [r.get(f"lm{j}_z", 0.0) for j in range(33)]
        for ln, a, b in conn_lines:
            ln.set_data([xs[a], xs[b]], [zs[a], zs[b]])
            ln.set_3d_properties([ys[a], ys[b]])
        arm_line.set_data([xs[j] for j in _ARM_CHAIN], [zs[j] for j in _ARM_CHAIN])
        arm_line.set_3d_properties([ys[j] for j in _ARM_CHAIN])
        scat._offsets3d = (xs, zs, ys)
        title.set_text(f"Frame {r['frame']}")
        return [ln for ln, _, _ in conn_lines] + [arm_line, title]

    _anim = animation.FuncAnimation(fig, update, frames=len(records), interval=50, blit=False)
    # keep reference so GC doesn't collect it
    fig._anim = _anim
    if save_dir:
        _anim.save(os.path.join(save_dir, "pose_3d.gif"), writer="pillow", fps=20)


def plot_compare(fk_records: list[ArmRecord], ik_records: list[ArmRecord],
                 save_dir: str | None = None, headless: bool = False) -> None:
    if not fk_records or not ik_records:
        return
    if save_dir is None:
        save_dir = _make_run_dir()
    n = min(len(fk_records), len(ik_records))

    def extract_chain(records):
        return [
            [[r[f"joint{j+1}_{ax}"] for ax in ("x", "y", "z")] for j in range(7)]
            for r in records[:n]
        ]

    fk_chain = extract_chain(fk_records)
    ik_chain = extract_chain(ik_records)

    all_pts = [pt for chain in (fk_chain, ik_chain) for frame in chain for pt in frame]
    xlim = (min(p[0] for p in all_pts), max(p[0] for p in all_pts))
    ylim = (min(p[1] for p in all_pts), max(p[1] for p in all_pts))
    zlim = (min(p[2] for p in all_pts), max(p[2] for p in all_pts))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    title = ax.set_title("Frame 0: FK vs IK arm chain")

    fk_line, = ax.plot([], [], [], 'b-o', lw=2, label="FK")
    ik_line, = ax.plot([], [], [], 'o-', color='orange', lw=2, label="IK")
    ax.legend()
    ax.view_init(elev=20, azim=-60)

    def update(i):
        for line, chain in [(fk_line, fk_chain), (ik_line, ik_chain)]:
            pts = chain[i]
            line.set_data([p[0] for p in pts], [p[1] for p in pts])
            line.set_3d_properties([p[2] for p in pts])
        title.set_text(f"Frame {fk_records[i]['frame']}: FK vs IK arm chain")
        return fk_line, ik_line, title

    _anim = animation.FuncAnimation(fig, update, frames=n, interval=50, blit=False)
    fig._anim = _anim
    _anim.save(os.path.join(save_dir, "compare_fk_ik.gif"), writer="pillow", fps=20)
    if not headless:
        plt.show(block=True)
    plt.close('all')
    logger.info("Compare arm chain saved to %s", save_dir)


def plot_compare_eef(fk_records: list[ArmRecord], ik_records: list[ArmRecord],
                     save_dir: str | None = None, headless: bool = False) -> None:
    """Plot EEF position (x,y,z) and orientation (alpha,beta,gamma) over time for FK vs IK."""
    if not fk_records or not ik_records:
        return
    if save_dir is None:
        save_dir = _make_run_dir()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("End-Effector: FK vs IK")

    pos_labels = [("eef_x", "X"), ("eef_y", "Y"), ("eef_z", "Z")]
    ori_labels = [("eef_alpha", "Alpha"), ("eef_beta", "Beta"), ("eef_gamma", "Gamma")]

    fk_frames = [r["frame"] for r in fk_records]
    ik_frames = [r["frame"] for r in ik_records]

    for col, (key, label) in enumerate(pos_labels):
        ax = axes[0, col]
        ax.plot(fk_frames, [r[key] for r in fk_records], color='blue', label="FK")
        ax.plot(ik_frames, [r[key] for r in ik_records], color='orange', label="IK")
        ax.set_title(f"Position {label}")
        ax.set_ylabel("Position (m)")
        ax.set_xlabel("Frame")
        ax.legend()
        ax.grid(True)

    for col, (key, label) in enumerate(ori_labels):
        ax = axes[1, col]
        ax.plot(fk_frames, [r[key] for r in fk_records], color='blue', label="FK")
        ax.plot(ik_frames, [r[key] for r in ik_records], color='orange', label="IK")
        ax.set_title(f"Orientation {label}")
        ax.set_ylabel("Angle (rad)")
        ax.set_xlabel("Frame")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "compare_eef.png"), dpi=150, bbox_inches="tight")
    if not headless:
        plt.show(block=True)
    plt.close('all')
    logger.info("Compare EEF plot saved to %s", save_dir)


def plot_all(records: list[ArmRecord], active_joints: set[int] | None = None,
             save_dir: str | None = None, headless: bool = False) -> None:
    if save_dir is None:
        save_dir = make_run_dir()
    for fn in [
        lambda: plot_arm_tracking(records, active_joints, save_dir=save_dir),
        lambda: plot_human_angles(records, save_dir=save_dir),
        lambda: plot_robot_state(records, save_dir=save_dir),
        lambda: plot_pose_3d_animated(records, save_dir=save_dir),
    ]:
        fn()
        if not headless:
            plt.show(block=True)
        plt.close('all')
    logger.info("Plots saved to %s", save_dir)
