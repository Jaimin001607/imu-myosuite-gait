import numpy as np
import pandas as pd
from io import StringIO
from scipy.signal import butter, filtfilt


MOT_PATH = "/Users/"Users name"/Downloads/subject01_walk1_ik.mot"

# These should match your MuJoCo joint names for MyoLeg:
JOINTS = [
    # Right
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    # Left
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
]

# Sign flips if needed (start all +1; flip any joint that moves wrong)
SIGN = {jn: 1.0 for jn in JOINTS}

# Example if knees bend backwards:
# SIGN["knee_angle_r"] = -1.0
# SIGN["knee_angle_l"] = -1.0

# Offsets: subtract mean of first N frames to match model neutral pose
OFFSET_N = 20

# Filtering (remove jitter before differentiating)
USE_FILTER = True
CUTOFF_HZ = 6.0
FILTER_ORDER = 2


def load_mot(path: str) -> pd.DataFrame:
    """OpenSim .mot: header until 'endheader', then whitespace table."""
    lines = open(path, "r", errors="ignore").read().splitlines()
    try:
        end = lines.index("endheader")
    except ValueError:
        raise RuntimeError("Could not find 'endheader' in .mot file.")
    table = "\n".join(lines[end + 1 :]).strip()
    df = pd.read_csv(StringIO(table), sep=r"\s+|\t", engine="python")
    if "time" not in df.columns:
        raise RuntimeError("No 'time' column in .mot.")
    return df


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    if len(x) < 10:
        return x
    nyq = 0.5 * fs
    wn = min(max(cutoff / nyq, 1e-4), 0.9999)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def central_diff(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central difference with endpoints one-sided."""
    dydt = np.empty_like(y)
    dt = np.diff(t)
    # endpoints
    dydt[0] = (y[1] - y[0]) / (t[1] - t[0])
    dydt[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    # interior
    dydt[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    return dydt


def main():
    df = load_mot(MOT_PATH)

    t = df["time"].astype(float).to_numpy()
    duration = t[-1] - t[0]
    dt_med = float(np.median(np.diff(t)))
    fs = 1.0 / dt_med

    print("Rows:", len(df), "Cols:", len(df.columns))
    print(f"Duration: {duration:.3f} s | dt_med: {dt_med:.6f} s | fs~{fs:.2f} Hz")

    missing = [jn for jn in JOINTS if jn not in df.columns]
    if missing:
        print("Missing joints in .mot:", missing)
        raise SystemExit(1)

    # Build q (rad) in a fixed order: JOINTS
    Q = np.zeros((len(t), len(JOINTS)), dtype=float)

    for j, jn in enumerate(JOINTS):
        ang_deg = df[jn].astype(float).to_numpy()
        ang_rad = np.deg2rad(ang_deg) * SIGN.get(jn, 1.0)

        # offset to neutral (first frames)
        ang_rad = ang_rad - float(np.mean(ang_rad[:OFFSET_N]))

        # filter
        if USE_FILTER:
            ang_rad = lowpass(ang_rad, fs, CUTOFF_HZ, FILTER_ORDER)

        Q[:, j] = ang_rad

    # qdot from filtered q
    QDOT = np.zeros_like(Q)
    for j in range(Q.shape[1]):
        QDOT[:, j] = central_diff(Q[:, j], t)

    # Diagnostics: per joint ranges
    print("\nJoint ranges (deg, after offset/filter):")
    for j, jn in enumerate(JOINTS):
        lo = np.rad2deg(Q[:, j].min())
        hi = np.rad2deg(Q[:, j].max())
        print(f"  {jn:16s}  [{lo:8.2f}, {hi:8.2f}]")

    out = "/Users/jaiminsuthar/Downloads/subject01_walk1_q_qdot.npz"
    np.savez(
        out,
        t=t,
        joints=np.array(JOINTS, dtype=object),
        q=Q,
        q_dot=QDOT,
    )
    print("\nSaved:", out)
    print("Keys: t, joints, q, q_dot")


if __name__ == "__main__":
    main()
