import argparse
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

# EEG Preprocessing
FS = 125          # sampling rate (Hz)
NOTCH_FREQ = 50   # line noise frequency (Hz)
Q = 30            # notch quality factor
BP = (1, 40)      # band-pass limits (Hz)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype="band")

# Apply notch and band-pass filtering to raw EEG window.
# eeg_window: np.ndarray shape (n_samples, n_channels)
# returns: (n_samples, n_channels) float32

def preprocess(eeg_window: np.ndarray):
  
    # transpose to (channels, time)
    x = eeg_window.T.astype(np.float32)
    # notch filter
    b_n, a_n = signal.iirnotch(NOTCH_FREQ, Q, FS)
    x = signal.filtfilt(b_n, a_n, x, axis=1)
    # band-pass filter
    b_bp, a_bp = butter_bandpass(BP[0], BP[1], FS)
    x = signal.filtfilt(b_bp, a_bp, x, axis=1)
    # transpose back to (time, channels)
    return x.T

# Conversion Logic 
def convert_session(csv_path: Path, out_dir: Path, expected_len: int = None):
    df = pd.read_csv(csv_path)
    ch_cols = [c for c in df.columns if c.startswith('ch')]
    labels = df['label'].astype(int)
    unique_labels = labels.drop_duplicates().tolist()

    # Determine expected length from first session if not provided
    if expected_len is None:
        first_lbl = unique_labels[0]
        expected_len = int((labels == first_lbl).sum())

    windows, window_labels = [], []
    for lbl in unique_labels:
        chunk = df[labels == lbl]
        n = len(chunk)
        raw = chunk[ch_cols].values  # (n, 16)
        proc = preprocess(raw)       # filtered (n, 16)

        if n > expected_len:
            print(f"⚠️  Truncating label {lbl} from {n} → {expected_len} samples")
            proc = proc[:expected_len]
        elif n < expected_len:
            print(f"⚠️  Padding label {lbl} from {n} → {expected_len} samples")
            pad = np.zeros((expected_len - n, len(ch_cols)), dtype=np.float32)
            proc = np.vstack([proc, pad])

        # transpose to (16, expected_len)
        windows.append(proc.T)
        window_labels.append(lbl)

    eeg = np.stack(windows)                 # (n_windows, 16, expected_len)
    labels_arr = np.array(window_labels, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(out_path, eeg=eeg, labels=labels_arr)
    print(f"Saved {out_path} → eeg {eeg.shape}, labels {labels_arr.shape}")

    return expected_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=Path, required=True,
                        help='Directory containing session_*.csv')
    parser.add_argument('--out_dir', type=Path, required=True,
                        help='Directory to save .npz files')
    args = parser.parse_args()

    exp_len = None
    for csv_file in sorted(Path(args.csv_dir).glob('session_*.csv')):
        exp_len = convert_session(csv_file, Path(args.out_dir), expected_len=exp_len)