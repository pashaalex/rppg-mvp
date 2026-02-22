"""
Plot phone vs photo signals on the same time axis (seconds).

Reads:
  - signals_phone.csv (fps=29.99)
  - signals_photo.csv (fps=50)

Columns expected:
  frame, face_g_mean, ref_g_mean

What it does:
  - converts frame -> time (seconds)
  - shifts both timelines to start at t=0
  - plots phone and photo on the same axes (optionally also ref)
  - z-scores each curve to make amplitudes comparable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------- CONFIG -----------------
PHONE_SIGNALS_CSV = "signals_phone.csv"
PHOTO_SIGNALS_CSV = "signals_photo.csv"

FPS_PHONE = 29.99
FPS_PHOTO = 50.0

FACE_COL = "face_g_mean"
REF_COL = "ref_g_mean"

PLOT_SECONDS = 30.0       # None -> plot full overlap
PLOT_REF = False           # also plot ref_g_mean as dashed lines
DOWNSAMPLE_MAX_POINTS = 20000  # to keep plotting responsive; None to disable
# ------------------------------------------


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.mean(x)
    s = np.std(x)
    if not np.isfinite(s) or s <= 0:
        return x * 0.0
    return (x - m) / s


def load_signal(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"frame", col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    out = df[["frame", col]].copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna()
    return out


def frame_to_time(df: pd.DataFrame, fps: float) -> np.ndarray:
    return df["frame"].to_numpy(dtype=np.float64) / float(fps)


def maybe_downsample(t: np.ndarray, y: np.ndarray, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or t.size <= max_points:
        return t, y
    idx = np.linspace(0, t.size - 1, max_points).astype(int)
    return t[idx], y[idx]


def main():
    # Load phone
    phone_face = load_signal(PHONE_SIGNALS_CSV, FACE_COL)
    t_phone = frame_to_time(phone_face, FPS_PHONE)
    y_phone_face = zscore(phone_face[FACE_COL].to_numpy(dtype=np.float64))

    # Load photo
    photo_face = load_signal(PHOTO_SIGNALS_CSV, FACE_COL)
    t_photo = frame_to_time(photo_face, FPS_PHOTO)
    y_photo_face = zscore(photo_face[FACE_COL].to_numpy(dtype=np.float64))

    # Shift both to start at t=0
    t_phone = t_phone - float(t_phone.min())
    t_photo = t_photo - float(t_photo.min())

    # Optional: limit to common duration (and/or PLOT_SECONDS)
    t_end = min(float(t_phone.max()), float(t_photo.max()))
    if PLOT_SECONDS is not None:
        t_end = min(t_end, float(PLOT_SECONDS))

    m_phone = (t_phone >= 0) & (t_phone <= t_end)
    m_photo = (t_photo >= 0) & (t_photo <= t_end)

    t_phone_p = t_phone[m_phone]
    y_phone_face_p = y_phone_face[m_phone]

    t_photo_p = t_photo[m_photo]
    y_photo_face_p = y_photo_face[m_photo]

    t_phone_p, y_phone_face_p = maybe_downsample(t_phone_p, y_phone_face_p, DOWNSAMPLE_MAX_POINTS)
    t_photo_p, y_photo_face_p = maybe_downsample(t_photo_p, y_photo_face_p, DOWNSAMPLE_MAX_POINTS)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t_phone_p, y_phone_face_p, label=f"Phone {FACE_COL} (fps={FPS_PHONE:.2f})")
    plt.plot(t_photo_p, y_photo_face_p, label=f"Photo {FACE_COL} (fps={FPS_PHOTO:.2f})")

    # Optional: ref signals (also z-scored independently)
    if PLOT_REF:
        phone_ref = load_signal(PHONE_SIGNALS_CSV, REF_COL)
        t_phone_r = frame_to_time(phone_ref, FPS_PHONE) - float(frame_to_time(phone_ref, FPS_PHONE).min())
        y_phone_ref = zscore(phone_ref[REF_COL].to_numpy(dtype=np.float64))
        mrp = (t_phone_r >= 0) & (t_phone_r <= t_end)
        t_phone_rp = t_phone_r[mrp]
        y_phone_refp = y_phone_ref[mrp]
        t_phone_rp, y_phone_refp = maybe_downsample(t_phone_rp, y_phone_refp, DOWNSAMPLE_MAX_POINTS)
        plt.plot(t_phone_rp, y_phone_refp, linestyle="--", label=f"Phone {REF_COL} (z)")

        photo_ref = load_signal(PHOTO_SIGNALS_CSV, REF_COL)
        t_photo_r = frame_to_time(photo_ref, FPS_PHOTO) - float(frame_to_time(photo_ref, FPS_PHOTO).min())
        y_photo_ref = zscore(photo_ref[REF_COL].to_numpy(dtype=np.float64))
        mrc = (t_photo_r >= 0) & (t_photo_r <= t_end)
        t_photo_rc = t_photo_r[mrc]
        y_photo_refc = y_photo_ref[mrc]
        t_photo_rc, y_photo_refc = maybe_downsample(t_photo_rc, y_photo_refc, DOWNSAMPLE_MAX_POINTS)
        plt.plot(t_photo_rc, y_photo_refc, linestyle="--", label=f"Photo {REF_COL} (z)")

    plt.title("Phone vs Photo signals on the same time axis (z-scored)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
