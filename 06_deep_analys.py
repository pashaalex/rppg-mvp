# compare_rppg_with_ecg_marks_from_filenames.py

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# --------
SIGNALS_CSV = "signals.csv"
MARKS_CSV   = "mark.csv"
OUT_DIR = Path("FINAL_COMPARISON")

FPS = 50.0

EPS = 1.0
DETREND_WIN = 251          # about 5s at 50 fps

# peak detection
MIN_PEAK_DISTANCE_SEC = 0.4    # about 150 bpm max
PEAK_PROMINENCE = 0.5          # after z-score

# matching tolerance around each R-peak
MATCH_TOL_FRAMES = int(0.25 * FPS)   # +/- 250 ms
# ---------


def rolling_median_detrend(x, win):
    s = pd.Series(x)
    return (s - s.rolling(win, center=True, min_periods=20).median()).to_numpy()


def build_pos_signal(df):
    fr, fg, fb = df["face_r_mean"], df["face_g_mean"], df["face_b_mean"]
    rr, rg, rb = df["ref_r_mean"],  df["ref_g_mean"],  df["ref_b_mean"]

    xr = np.log(fr + EPS) - np.log(rr + EPS)
    xg = np.log(fg + EPS) - np.log(rg + EPS)
    xb = np.log(fb + EPS) - np.log(rb + EPS)

    dr = rolling_median_detrend(xr, DETREND_WIN)
    dg = rolling_median_detrend(xg, DETREND_WIN)
    db = rolling_median_detrend(xb, DETREND_WIN)

# POS projections (Wang et al., IEEE TBME 2017)
# Vectors are orthogonal to illumination direction (1,1,1)
    X = 3*dr - 2*dg
    Y = 1.5*dr + dg - 1.5*db
    alpha = np.nanstd(X) / (np.nanstd(Y) + 1e-9)

    pos = X - alpha*Y
    pos = (pos - np.nanmean(pos)) / (np.nanstd(pos) + 1e-9)
    return pos


def parse_frame_from_filename(fname: str) -> int:
    """
    Accepts:
      '00042.png' -> 42
      'frame_00042.png' -> 42
      '.../00042.png' -> 42
    """
    base = Path(fname).name
    m = re.search(r"(\d+)", base)
    if not m:
        raise ValueError(f"Cannot parse frame number from filename: {fname}")
    return int(m.group(1))


def match_peaks(r_peaks_idx, ppg_peaks_idx, tol):
    """
    r_peaks_idx and ppg_peaks_idx are indices in the merged dataframe (NOT frame numbers).
    """
    matches = []
    used = set()
    for r in r_peaks_idx:
        candidates = [(p, abs(p - r)) for p in ppg_peaks_idx if abs(p - r) <= tol and p not in used]
        if candidates:
            p, d = min(candidates, key=lambda x: x[1])
            matches.append((r, p, p - r))
            used.add(p)
    return matches

def peak_analys(peaks):
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 3:
        return np.nan
    rr = np.diff(peaks)
    cv = np.std(rr, ddof=1) / np.mean(rr)
    bpm = FPS * 60.0 / np.mean(rr)
    print(f"Coefficient of variation for BPM = {cv:0.3f}")
    print(f"AVG(BPM)={bpm:0.3f}")

def main():
    OUT_DIR.mkdir(exist_ok=True)

    sig = pd.read_csv(SIGNALS_CSV)
    marks = pd.read_csv(MARKS_CSV)

    # ---- normalize signals ----
    if "frame" not in sig.columns:
        # fallback: assume row order is frame
        sig = sig.copy()
        sig["frame"] = np.arange(len(sig), dtype=int)

    sig["frame"] = pd.to_numeric(sig["frame"], errors="coerce")
    sig = sig.dropna(subset=["frame"]).copy()
    sig["frame"] = sig["frame"].astype(int)

    marks = marks.copy()
    marks["frame"] = marks["file"].astype(str).map(parse_frame_from_filename)
    marks["mark"] = pd.to_numeric(marks["mark"], errors="coerce").fillna(0).astype(int)

    # Some mark.csv have frames starting at 0, while signals.csv might start at e.g. 2090.
    # We'll merge strictly on frame values that exist in both.
    df = sig.merge(marks[["frame", "mark"]], on="frame", how="inner").sort_values("frame").reset_index(drop=True)

    frames = df["frame"].to_numpy(dtype=int)

    # Build rPPG
    required_cols = ["face_r_mean", "face_g_mean", "face_b_mean", "ref_r_mean", "ref_g_mean", "ref_b_mean"]
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    frames = df["frame"].to_numpy(dtype=int)

    pos = build_pos_signal(df)

    # Detect rPPG peaks (indices in df)
    min_dist_frames = int(MIN_PEAK_DISTANCE_SEC * FPS)
    ppg_peaks_idx, _ = find_peaks(pos, distance=min_dist_frames, prominence=PEAK_PROMINENCE)
    print("PEAK RESULTS")
    peak_analys(ppg_peaks_idx)

    # ECG R-peaks (indices in df)
    r_peaks_idx = df.index[df["mark"] == 1].to_numpy()
    print("PEAK RESULTS")
    peak_analys(r_peaks_idx)

    # Match peaks with tolerance (in index units ~ frames, since frame increases by 1)
    # If your merged df has missing frames, this is slightly off; we handle it by matching in frame-values below.
    # Let's do matching in actual frame numbers to be safe:
    ppg_peak_frames = frames[ppg_peaks_idx]
    r_peak_frames = frames[r_peaks_idx]

    # Frame-based matching
    matches = []
    used_ppg = set()
    for r_fr in r_peak_frames:
        cand = [(p_fr, abs(p_fr - r_fr)) for p_fr in ppg_peak_frames if abs(p_fr - r_fr) <= MATCH_TOL_FRAMES and p_fr not in used_ppg]
        if cand:
            p_fr, d = min(cand, key=lambda x: x[1])
            matches.append((r_fr, p_fr, p_fr - r_fr))
            used_ppg.add(p_fr)

    # Metrics
    tp = len(matches)
    fp = len(ppg_peak_frames) - tp
    fn = len(r_peak_frames) - tp

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    lags = np.array([lag for _, _, lag in matches], dtype=float)

    metrics = {
        "n_frames_merged": len(df),
        "n_r_peaks": int(len(r_peak_frames)),
        "n_ppg_peaks": int(len(ppg_peak_frames)),
        "tp_matched": int(tp),
        "fp_ppg": int(fp),
        "fn_r": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "mean_lag_frames": float(lags.mean()) if len(lags) else float("nan"),
        "std_lag_frames": float(lags.std()) if len(lags) else float("nan"),
        "mean_lag_ms": float(1000.0 * lags.mean() / FPS) if len(lags) else float("nan"),
        "std_lag_ms": float(1000.0 * lags.std() / FPS) if len(lags) else float("nan"),
        "match_tolerance_frames": int(MATCH_TOL_FRAMES),
        "fps": float(FPS),
    }

    pd.Series(metrics).to_csv(OUT_DIR / "metrics.csv")

    # Save matches table
    if matches:
        pd.DataFrame(matches, columns=["r_frame", "ppg_frame", "lag_frames"]).to_csv(OUT_DIR / "matches.csv", index=False)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(frames, pos, label="rPPG (POS)", alpha=0.9, linewidth=1)

    # mark ppg peaks
    # map peak frames to indices for y-values
    peak_frame_to_idx = {f: i for i, f in enumerate(frames)}
    ppg_peak_idx2 = [peak_frame_to_idx[f] for f in ppg_peak_frames if f in peak_frame_to_idx]
    plt.scatter(frames[ppg_peak_idx2], pos[ppg_peak_idx2], color="red", s=25, label="rPPG peaks")

    # ECG R-peaks as vertical lines
    plt.vlines(r_peak_frames, ymin=np.nanmin(pos), ymax=np.nanmax(pos),
               color="green", alpha=0.25, label="ECG R-peaks (manual)")

    # annotate mean lag if present
    if len(lags):
        plt.text(frames[0], np.nanmax(pos)*0.85,
                 f"Mean lag: {lags.mean():0.2f} frames ({1000*lags.mean()/FPS:.0f} ms)\n"
                 f"Precision: {precision:0.2f}, Recall: {recall:0.2f}",
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title("rPPG vs manually annotated ECG R-peaks (frame axis)")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude (z)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rppg_vs_ecg.png", dpi=150)
    plt.close()

    print("=== RESULTS ===")
    for k in ["precision", "recall", "mean_lag_frames", "mean_lag_ms"]:
        print(f"{k}: {metrics[k]:0.3f}")

    print(f"Saved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
