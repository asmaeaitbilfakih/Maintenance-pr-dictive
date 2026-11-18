import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
BANDS_HZ = [(0, 60), (60, 200), (200, 500), (500, 1000)]   # bandes d'énergie FFT
 
 
# --------------------------------------------------------------------------- vibrations
def signal_features(x, fs):
    """Features d'UNE fenêtre de signal (tableau 1D)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    std = x.std() + 1e-12
    feats = {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": x.max() - x.min(),
        "crest_factor": peak / (rms + 1e-12),
        "kurtosis": np.mean((x / std) ** 4),          # 3 = signal gaussien sain
        "skewness": np.mean((x / std) ** 3),
    }
    # spectre (fenêtre de Hann pour limiter les fuites spectrales)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    power = spec ** 2
    total = power.sum() + 1e-12
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi)
        feats[f"band_{lo}_{hi}"] = power[m].sum() / total       # part d'énergie dans la bande
    feats["dominant_freq"] = freqs[np.argmax(power[1:]) + 1]
    feats["spectral_centroid"] = (freqs * power).sum() / total
    return feats
 
 
def signals_to_features(signals, index):
    """signals : (n, N) ; index : DataFrame aligné (colonne fs_hz). -> DataFrame de features."""
    rows = [signal_features(signals[i], index["fs_hz"].iloc[i]) for i in range(len(signals))]
    feats = pd.DataFrame(rows).add_prefix("sig_")
    return pd.concat([index.reset_index(drop=True), feats], axis=1)
 
 
# --------------------------------------------------------------------------- temporel
SENSOR_COLS = ["vib_rms", "vib_kurtosis", "vib_crest", "temperature", "pressure", "current"]
 
 
def _slope(window):
    """Pente de la droite de régression sur la fenêtre (tendance)."""
    t = np.arange(len(window))
    return np.polyfit(t, window, 1)[0] if len(window) > 1 else 0.0
 
 
def temporal_features(df, win=4, long_win=12, baseline_n=10, cols=SENSOR_COLS):
    """Features glissantes PAR machine ET par cycle (on ne mélange jamais deux cycles).
    win = 4 mesures = 24 h avec dt = 6 h. N'utilise que le passé : pas de fuite."""
    df = df.sort_values(["machine_id", "cycle_id", "time_h"]).copy()
    g = df.groupby(["machine_id", "cycle_id"], sort=False)
    out = {}
    for c in cols:
        col = g[c]
        out[f"{c}_mean_{win}"] = col.transform(lambda s: s.rolling(win, min_periods=1).mean())
        out[f"{c}_std_{win}"] = col.transform(
            lambda s: s.rolling(win, min_periods=2).std()).fillna(0)
        out[f"{c}_slope_{long_win}"] = col.transform(
            lambda s: s.rolling(long_win, min_periods=3).apply(_slope, raw=True)).fillna(0)
        # écart à la baseline : moyenne des `baseline_n` premières mesures du cycle
        base = col.transform(lambda s: s.expanding().mean().where(
            np.arange(len(s)) < baseline_n).ffill())
        out[f"{c}_ratio_baseline"] = df[c] / base
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
 
 
# --------------------------------------------------------------------------- split
def split_by_machine(df, test_size=0.25, seed=42):
    """Découpe par MACHINE (jamais au hasard sur les lignes d'une même machine)."""
    rng = np.random.default_rng(seed)
    ids = df["machine_id"].unique()
    test_ids = set(rng.choice(ids, size=max(1, int(len(ids) * test_size)), replace=False))
    is_test = df["machine_id"].isin(test_ids)
    return df[~is_test].copy(), df[is_test].copy()
 
 
# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/synthetic")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
 
    sensors = pd.read_csv(src / "sensors.csv")
    feats = temporal_features(sensors)
    feats.round(4).to_csv(out / "features_temporal.csv", index=False)
    print(f"features temporelles : {feats.shape} -> {out / 'features_temporal.csv'}")
 
    signals = np.load(src / "signals.npy")
    index = pd.read_csv(src / "signals_index.csv")
    sf = signals_to_features(signals, index)
    sf.round(5).to_csv(out / "features_vibration.csv", index=False)
    print(f"features vibration   : {sf.shape} -> {out / 'features_vibration.csv'}")
 
 
if __name__ == "__main__":
    main()
 