#!/usr/bin/env python3
"""
Run --compare on each video individually, capture stdout, parse timing,
and write results/batch_compare_<timestamp>.json.
"""
import glob, json, os, re, subprocess, sys
from datetime import datetime

VIDEOS_DIR = "./videos"
RESULTS_DIR = "./results"
PYTHON = sys.executable


def parse_timing(stdout: str) -> dict:
    fk = re.search(r"FK total compute time:\s*([\d.]+)s", stdout)
    ik = re.search(r"IK total compute time:\s*([\d.]+)s", stdout)
    frames_match = re.search(r"\[FK\] \[.+?\] processed (\d+) frames", stdout)
    return {
        "fk_compute_s": float(fk.group(1)) if fk else None,
        "ik_compute_s": float(ik.group(1)) if ik else None,
        "frames": int(frames_match.group(1)) if frames_match else None,
    }


def main():
    videos = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not videos:
        print(f"No videos found in {VIDEOS_DIR}"); return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for path in videos:
        name = os.path.basename(path)
        print(f"--- {name} ---")
        proc = subprocess.run(
            [PYTHON, "-m", "src.main", "--compare", "--headless", "--video", name],
            capture_output=True, text=True
        )
        stdout = proc.stdout
        stderr = proc.stderr
        print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)

        timing = parse_timing(stdout)
        saved = None
        if timing["fk_compute_s"] is not None and timing["ik_compute_s"] is not None:
            saved = timing["ik_compute_s"] - timing["fk_compute_s"]

        results.append({
            "video": name,
            "frames": timing["frames"],
            "fk_compute_s": timing["fk_compute_s"],
            "ik_compute_s": timing["ik_compute_s"],
            "time_saved_by_fk_s": saved,
            "stdout": stdout,
            "returncode": proc.returncode,
        })

    out_path = os.path.join(
        RESULTS_DIR, f"batch_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print("\n=== Batch Compare Summary ===")
    print(f"{'Video':<30} {'Frames':>7} {'FK (s)':>8} {'IK (s)':>8} {'Saved (s)':>10}")
    print("-" * 67)
    for r in results:
        print(f"{r['video']:<30} {str(r['frames'] or '?'):>7}"
              f" {str(round(r['fk_compute_s'], 4) if r['fk_compute_s'] else '?'):>8}"
              f" {str(round(r['ik_compute_s'], 4) if r['ik_compute_s'] else '?'):>8}"
              f" {str(round(r['time_saved_by_fk_s'], 4) if r['time_saved_by_fk_s'] is not None else '?'):>10}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
