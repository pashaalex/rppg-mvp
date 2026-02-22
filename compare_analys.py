import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend


# ---------------- CONFIG ----------------
PHONE_SIGNALS_CSV = "signals_phone.csv"
PHOTO_SIGNALS_CSV = "signals_photo.csv"

FPS_PHONE = 29.99
FPS_PHOTO = 50.0

WELCH_WINDOW_SEC = 8.0
WELCH_OVERLAP = 0.5

MAX_FREQ = 5.0
# ----------------------------------------


def compute_psd(x: np.ndarray, fs: float):
    x = detrend(x, type="linear")

    nperseg = int(WELCH_WINDOW_SEC * fs)
    noverlap = int(WELCH_OVERLAP * nperseg)

    f, pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )
    return f, pxx


def load_signal(path, col):
    df = pd.read_csv(path)
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    return x


def main():
    # --- load signals ---
    phone_face = load_signal(PHONE_SIGNALS_CSV, "face_g_mean")
    phone_ref  = load_signal(PHONE_SIGNALS_CSV, "ref_g_mean")

    photo_face = load_signal(PHOTO_SIGNALS_CSV, "face_g_mean")
    photo_ref  = load_signal(PHOTO_SIGNALS_CSV, "ref_g_mean")

    # --- PSD ---
    f_pf, psd_pf = compute_psd(phone_face, FPS_PHONE)
    f_pr, psd_pr = compute_psd(phone_ref,  FPS_PHONE)

    f_cf, psd_cf = compute_psd(photo_face, FPS_PHOTO)
    f_cr, psd_cr = compute_psd(photo_ref,  FPS_PHOTO)

    # --- plot ---
    plt.figure(figsize=(10, 6))

    plt.plot(f_pf, psd_pf, label="Phone – face_g")
    plt.plot(f_pr, psd_pr, label="Phone – ref_g", linestyle="--")

    plt.plot(f_cf, psd_cf, label="Photo – face_g")
    plt.plot(f_cr, psd_cr, label="Photo – ref_g", linestyle="--")

    plt.xlim(0, MAX_FREQ)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (linear)")
    plt.title("Welch PSD (linear scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.5, 3.0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
