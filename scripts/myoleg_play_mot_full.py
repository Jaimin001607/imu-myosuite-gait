import time
import numpy as np
import pandas as pd
from io import StringIO
from scipy.signal import butter, filtfilt
import mujoco
import mujoco.viewer

# --------------------------
# PATHS (EDIT THESE)
# --------------------------
MOT_PATH = "/Users/"Users name"/Downloads/subject01_walk1_ik.mot"
XML_PATH = "/Users/"Users name"/myo_sim/leg/myolegs.xml"

# --------------------------
# JOINTS TO DRIVE (MyoLeg)
# --------------------------
JOINTS = [
    # Right
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    # Left
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
]

# --------------------------
# CALIBRATION CONTROLS
# --------------------------
# Start all +1. If a joint bends wrong direction, flip to -1.
SIGN = {jn: 1.0 for jn in JOINTS}

# Typical fix if knees bend backwards in your model:
# SIGN["knee_angle_r"] = -1.0
# SIGN["knee_angle_l"] = -1.0

# Offset to align the motion to your model's neutral pose
OFFSET_N = 20  # subtract mean of first N frames

# Filtering (remove jitter)
USE_FILTER = True
CUTOFF_HZ = 6.0       # 4–8 Hz typical for gait angles
FILTER_ORDER = 2

# Playback speed: 1.0 = realtime, 0.5 = half speed (slower)
SPEED = 1.0

# Seamless loop blend duration (seconds)
BLEND_SEC = 0.10


# --------------------------
# HELPERS
# --------------------------
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


def mj_joint_qposadr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint '{joint_name}' not found in model.")
    return int(model.jnt_qposadr[jid])


def mj_joint_range(model: mujoco.MjModel, joint_name: str):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None, None
    low, high = model.jnt_range[jid]
    if low == 0.0 and high == 0.0:
        return None, None
    return float(low), float(high)


def make_seamless_loop(traj: dict, dt: float, blend_sec: float):
    """Crossfade last K frames into first K frames for each joint trajectory."""
    K = int(blend_sec / dt)
    K = max(5, min(K, len(next(iter(traj.values()))) // 10))
    w = np.linspace(0.0, 1.0, K)

    for jn, a in traj.items():
        start = a[:K].copy()
        end = a[-K:].copy()
        a[:K] = (1 - w) * end + w * start
        traj[jn] = a
    return traj


def main():
    # ---- Load .mot ----
    df = load_mot(MOT_PATH)
    t = df["time"].astype(float).to_numpy()
    duration = t[-1] - t[0]
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt

    print(f"Loaded .mot rows={len(t)} duration={duration:.3f}s dt={dt:.6f}s fs~{fs:.2f}Hz")

    missing = [jn for jn in JOINTS if jn not in df.columns]
    if missing:
        print("Missing columns in .mot:", missing)
        raise SystemExit(1)

    # ---- Load MuJoCo model ----
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # ---- qpos indices for joints ----
    qpos_idx = {jn: mj_joint_qposadr(model, jn) for jn in JOINTS}

    # ---- Root stabilization (prevents wobble/drift) ----
    ROOT_NAME = "root"
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_NAME)
    if root_jid < 0:
        print("[WARN] No joint named 'root' found. Root locking disabled.")
        root_qpos_start = None
        root_dof = 0
        root_qpos0 = None
    else:
        root_qpos_start = int(model.jnt_qposadr[root_jid])

        # Estimate root dof by finding next joint qpos address
        if root_jid < model.njnt - 1:
            root_qpos_end = int(model.jnt_qposadr[root_jid + 1])
        else:
            root_qpos_end = root_qpos_start

        root_dof = max(0, root_qpos_end - root_qpos_start)
        root_qpos0 = data.qpos[root_qpos_start : root_qpos_start + root_dof].copy()
        print(f"Root lock enabled: qpos[{root_qpos_start}:{root_qpos_start+root_dof}]")

    # ---- Build trajectories (rad) ----
    traj = {}
    for jn in JOINTS:
        ang = np.deg2rad(df[jn].astype(float).to_numpy())
        ang = SIGN.get(jn, 1.0) * ang

        # offset
        ang = ang - float(np.mean(ang[:OFFSET_N]))

        # filter
        if USE_FILTER:
            ang = lowpass(ang, fs, CUTOFF_HZ, FILTER_ORDER)

        # clamp to model joint limits if present
        low, high = mj_joint_range(model, jn)
        if low is not None and high is not None:
            ang = np.clip(ang, low, high)

        traj[jn] = ang

    # ---- Seamless looping to avoid snap ----
    traj = make_seamless_loop(traj, dt, BLEND_SEC)

    # ---- Diagnostics: ranges ----
    print("\nJoint ranges after processing (deg):")
    for jn in JOINTS:
        lo = np.rad2deg(traj[jn].min())
        hi = np.rad2deg(traj[jn].max())
        print(f"  {jn:16s} [{lo:8.2f}, {hi:8.2f}]")

    # ---- Playback loop ----
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = t[0]
        k = 0
        start = time.time()

        while viewer.is_running():
            # Lock root pose each frame to prevent drift (critical for wobble fix)
            if root_dof > 0:
                data.qpos[root_qpos_start : root_qpos_start + root_dof] = root_qpos0

            # Set joint qpos from trajectories
            for jn in JOINTS:
                data.qpos[qpos_idx[jn]] = traj[jn][k]

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Real-time pacing (scaled)
            target = start + ((t[k] - t0) / SPEED)
            now = time.time()
            if target > now:
                time.sleep(target - now)

            # advance and loop
            k += 1
            if k >= len(t):
                k = 0
                start = time.time()


if __name__ == "__main__":
    main()

