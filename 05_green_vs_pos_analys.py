# plot_green_vs_pos_with_marks_and_ecg_bpm.py
# Two-panel plot (Green-only + POS) with:
#  - detected rPPG peaks
#  - manual ECG R-peaks (vertical lines)
#  - mean BPM from Green peaks, POS peaks, and ECG marks

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------------- CONFIG ----------------
SIGNALS_CSV = "signals.csv"
MARKS_CSV   = "mark.csv"
OUT_PNG     = "comparison_green_pos.png"

FPS = 50.0
EPS = 1.0

DETREND_WIN = 251                 # ~5 sec at 50 fps
MIN_PEAK_DISTANCE_SEC = 0.4       # max ~150 bpm
PEAK_PROMINENCE = 0.5             # after z-score
# --------------------------------------


def rolling_median_detrend(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    med = s.rolling(window=win, center=True, min_periods=20).median()
    return (s - med).to_numpy(dtype=np.float64)


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def parse_frame_from_filename(fname: str) -> int:
    base = Path(fname).name
    m = re.search(r"(\d+)", base)
    if not m:
        raise ValueError(f"Cannot parse frame number from filename: {fname}")
    return int(m.group(1))


def build_green_signal(df: pd.DataFrame) -> np.ndarray:
    face_g = df["face_g_mean"].to_numpy(np.float64)
    ref_g  = df["ref_g_mean"].to_numpy(np.float64)

    xg = np.log(face_g + EPS) - np.log(ref_g + EPS)
    dg = rolling_median_detrend(xg, DETREND_WIN)
    return zscore(dg)


def build_pos_signal(df: pd.DataFrame) -> np.ndarray:
    fr = df["face_r_mean"].to_numpy(np.float64)
    fg = df["face_g_mean"].to_numpy(np.float64)
    fb = df["face_b_mean"].to_numpy(np.float64)

    rr = df["ref_r_mean"].to_numpy(np.float64)
    rg = df["ref_g_mean"].to_numpy(np.float64)
    rb = df["ref_b_mean"].to_numpy(np.float64)

    xr = np.log(fr + EPS) - np.log(rr + EPS)
    xg = np.log(fg + EPS) - np.log(rg + EPS)
    xb = np.log(fb + EPS) - np.log(rb + EPS)

    dr = rolling_median_detrend(xr, DETREND_WIN)
    dg = rolling_median_detrend(xg, DETREND_WIN)
    db = rolling_median_detrend(xb, DETREND_WIN)

    X = 3 * dr - 2 * dg
    Y = 1.5 * dr + 1.0 * dg - 1.5 * db

    alpha = np.nanstd(X) / (np.nanstd(Y) + 1e-9)
    pos = X - alpha * Y
    return zscore(pos)


def detect_peaks(sig_z: np.ndarray) -> np.ndarray:
    min_dist_frames = int(MIN_PEAK_DISTANCE_SEC * FPS)
    peaks_idx, _ = find_peaks(sig_z, distance=min_dist_frames, prominence=PEAK_PROMINENCE)
    return peaks_idx


def mean_bpm_from_peak_frames(peak_frames: np.ndarray) -> float:
    if len(peak_frames) < 2:
        return float("nan")
    #rr_frames = np.diff(peak_frames).astype(np.float64)
    #rr_sec = rr_frames / FPS
    #bpm = 60.0 / rr_sec
    rr = np.diff(peak_frames).astype(np.float64).mean()    
    #return float(np.mean(bpm))
    return 60.0 * FPS / rr


def main():
    sig = pd.read_csv(SIGNALS_CSV)
    marks = pd.read_csv(MARKS_CSV)

    # signals
    if "frame" not in sig.columns:
        sig = sig.copy()
        sig["frame"] = np.arange(len(sig), dtype=int)
    sig["frame"] = pd.to_numeric(sig["frame"], errors="coerce")
    sig = sig.dropna(subset=["frame"]).copy()
    sig["frame"] = sig["frame"].astype(int)

    # marks: file,mark -> frame,mark
    if "file" not in marks.columns or "mark" not in marks.columns:
        raise RuntimeError(f"mark.csv must have columns file,mark. Found: {list(marks.columns)}")
    marks = marks.copy()
    marks["frame"] = marks["file"].astype(str).map(parse_frame_from_filename)
    marks["mark"] = pd.to_numeric(marks["mark"], errors="coerce").fillna(0).astype(int)

    # merge
    df = sig.merge(marks[["frame", "mark"]], on="frame", how="inner").sort_values("frame").reset_index(drop=True)

    # ensure numeric for needed cols
    needed = ["face_r_mean","face_g_mean","face_b_mean","ref_r_mean","ref_g_mean","ref_b_mean"]
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed).reset_index(drop=True)

    frames = df["frame"].to_numpy(dtype=int)
    r_peak_frames = df.loc[df["mark"] == 1, "frame"].to_numpy(dtype=int)

    # ECG mean BPM (ground truth)
    ecg_mean_bpm = mean_bpm_from_peak_frames(r_peak_frames)

    # signals
    green = build_green_signal(df)
    pos   = build_pos_signal(df)

    # peaks
    green_peaks_idx = detect_peaks(green)
    pos_peaks_idx   = detect_peaks(pos)

    green_peak_frames = frames[green_peaks_idx]
    pos_peak_frames   = frames[pos_peaks_idx]

    green_mean_bpm = mean_bpm_from_peak_frames(green_peak_frames)
    pos_mean_bpm   = mean_bpm_from_peak_frames(pos_peak_frames)

    print(f"ECG   mean BPM: {ecg_mean_bpm:.2f}")
    print(f"Green mean BPM: {green_mean_bpm:.2f}")
    print(f"POS   mean BPM: {pos_mean_bpm:.2f}")

    # plotting (two panels)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Green
    ax = axes[0]
    ax.plot(frames, green, linewidth=1, label="Green-only (log(face/ref), detrended, z)")
    ax.scatter(frames[green_peaks_idx], green[green_peaks_idx], s=22, label="Detected peaks")
    ax.vlines(r_peak_frames, ymin=np.nanmin(green), ymax=np.nanmax(green), alpha=0.25, label="Manual ECG R-peaks")
    ax.set_title(f"Green-only — mean BPM ≈ {green_mean_bpm:.2f} (ECG ≈ {ecg_mean_bpm:.2f})")
    ax.set_ylabel("Amplitude (z)")
    ax.legend(loc="upper right")

    # Panel 2: POS
    ax = axes[1]
    ax.plot(frames, pos, linewidth=1, label="POS (log(face/ref), detrended, z)")
    ax.scatter(frames[pos_peaks_idx], pos[pos_peaks_idx], s=22, label="Detected peaks")
    ax.vlines(r_peak_frames, ymin=np.nanmin(pos), ymax=np.nanmax(pos), alpha=0.25, label="Manual ECG R-peaks")
    ax.set_title(f"POS — mean BPM ≈ {pos_mean_bpm:.2f} (ECG ≈ {ecg_mean_bpm:.2f})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Amplitude (z)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
