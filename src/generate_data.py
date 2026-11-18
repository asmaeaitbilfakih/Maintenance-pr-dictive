import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FS_HZ = 2048          # fréquence d'échantillonnage du signal brut (Hz)
WINDOW = 1024         # échantillons par fenêtre (0,5 s)
F_ROT = {"ventilateur": 24.0, "broyeur": 15.0, "compresseur": 50.0}   # fréquence de rotation (Hz)
BPFO_RATIO = 3.58     # fréquence de défaut de bague externe = ratio x f_rot
F_RES = 650.0         # résonance de la structure excitée par les chocs (Hz)

MACHINE_TYPES = {
    # type: (beta Weibull, eta en heures, rms de base mm/s, temp de base °C, pression bar, courant A)
    "ventilateur": (2.6, 2400, 2.0, 55.0, 1.2, 45.0),
    "broyeur":     (2.2, 1800, 4.5, 70.0, 2.0, 120.0),
    "compresseur": (3.0, 3000, 2.8, 65.0, 7.5, 60.0),
}


def make_signal(rng, rms, hp, f_rot):
    """Une fenêtre de vibration. hp in [0,1] = degradation. Sain : 1x/2x + bruit.
    Dégradé : harmoniques plus fortes + chocs périodiques (défaut de roulement)."""
    t = np.arange(WINDOW) / FS_HZ
    ph = rng.uniform(0, 2 * np.pi, 3)
    x = (np.sin(2 * np.pi * f_rot * t + ph[0])
         + (0.5 + 1.5 * hp) * np.sin(2 * np.pi * 2 * f_rot * t + ph[1])
         + 1.0 * hp * np.sin(2 * np.pi * 3 * f_rot * t + ph[2])
         + 0.35 * rng.standard_normal(WINDOW))
    # chocs au rythme du défaut, filtrés par la résonance de la machine
    f_bpfo = BPFO_RATIO * f_rot * rng.uniform(0.99, 1.01)
    imp = np.zeros(WINDOW)
    pos = np.arange(rng.uniform(0, 1 / f_bpfo), WINDOW / FS_HZ, 1 / f_bpfo)
    imp[(pos * FS_HZ).astype(int) % WINDOW] = 6.0 * hp * rng.uniform(0.6, 1.4, len(pos))
    k = np.arange(96) / FS_HZ
    kernel = np.exp(-k * 700) * np.sin(2 * np.pi * F_RES * k)
    x = x + np.convolve(imp, kernel)[:WINDOW]
    return (x * rms / x.std()).astype(np.float32)


def simulate_machine(machine_id, mtype, rng, horizon_h, dt_h, horizon_label_h, preventive_h,
                     signal_every=20):
    beta, eta, rms0, temp0, pres0, cur0 = MACHINE_TYPES[mtype]
    eta_m = eta * rng.lognormal(0, 0.15)            # chaque machine a son échelle
    load = rng.uniform(0.6, 1.0)                    # charge moyenne (covariable)
    sensor_rows, maint_rows, cycle_rows = [], [], []
    sig_list, sig_idx = [], []

    t, cycle = 0.0, 0
    while t < horizon_h:
        cycle += 1
        life = eta_m * rng.weibull(beta) / load      # durée de vie de ce cycle (h)
        t_end_planned = t + life
        # fin du cycle : panne, préventif ou fin d'observation (censure)
        t_prev = t + preventive_h if preventive_h else np.inf
        t_stop = min(t_end_planned, t_prev, horizon_h)
        if t_stop == t_end_planned:
            end_type = "panne"
        elif t_stop == t_prev:
            end_type = "preventive"
        else:
            end_type = "fin_observation"

        times = np.arange(t, t_stop, dt_h)
        if len(times):
            age = times - t
            h = np.clip(age / life, 0, 1)                    # indice de santé caché
            hp = h ** 3                                      # dégradation accélérée en fin de vie
            n = len(times)
            vib_rms = rms0 * (1 + 1.8 * hp) + rng.normal(0, 0.06 * rms0, n)
            vib_kurt = 3.0 + 4.0 * hp + rng.normal(0, 0.25, n)       # 3 = signal sain
            vib_crest = 3.2 + 2.2 * hp + rng.normal(0, 0.15, n)
            temp = temp0 + 18 * hp + 4 * load + rng.normal(0, 1.0, n)
            pres = pres0 * (1 - 0.15 * hp) + rng.normal(0, 0.03 * pres0, n)
            cur = cur0 * load * (1 + 0.25 * hp) + rng.normal(0, 0.02 * cur0, n)
            rul = life - age                                  # heures avant la fin du cycle
            df = pd.DataFrame({
                "machine_id": machine_id, "machine_type": mtype, "cycle_id": cycle,
                "time_h": times, "age_h": age,
                "vib_rms": vib_rms, "vib_kurtosis": vib_kurt, "vib_crest": vib_crest,
                "temperature": temp, "pressure": pres, "current": cur,
                "rul_h": np.where(end_type == "panne", rul, np.nan),  # RUL inconnue si censuré
                "load": load,
            })
            df["failure_within_h"] = (
                (end_type == "panne") & (rul <= horizon_label_h)
            ).astype(int)
            sensor_rows.append(df)
            if signal_every:
                for j in range(0, n, signal_every):
                    sig_list.append(make_signal(rng, vib_rms[j], hp[j], F_ROT[mtype]))
                    sig_idx.append({
                        "machine_id": machine_id, "machine_type": mtype, "cycle_id": cycle,
                        "time_h": times[j], "age_h": age[j], "rul_h": df.rul_h.iloc[j],
                        "failure_within_h": int(df.failure_within_h.iloc[j]),
                        "fs_hz": FS_HZ})

        duration = t_stop - t
        cycle_rows.append({
            "machine_id": machine_id, "machine_type": mtype, "cycle_id": cycle,
            "start_h": t, "duration_h": duration,
            "event": int(end_type == "panne"), "end_type": end_type, "load": load,
        })
        if end_type == "fin_observation":
            break
        downtime = rng.uniform(24, 72) if end_type == "panne" else rng.uniform(4, 12)
        maint_rows.append({
            "machine_id": machine_id, "time_h": t_stop, "type": (
                "corrective" if end_type == "panne" else "preventive"),
            "downtime_h": downtime,
        })
        t = t_stop + downtime
    return sensor_rows, maint_rows, cycle_rows, sig_list, sig_idx


def generate(n_machines=60, horizon_h=8000, dt_h=6, label_h=48, preventive_h=None, seed=42,
             signal_every=20):
    rng = np.random.default_rng(seed)
    types = list(MACHINE_TYPES)
    machines, sensors, maint, cycles = [], [], [], []
    sigs, sig_idx = [], []
    for i in range(n_machines):
        mid = f"M{i + 1:03d}"
        mtype = types[i % len(types)]
        s, m, c, sg, si = simulate_machine(mid, mtype, rng, horizon_h, dt_h, label_h,
                                           preventive_h, signal_every)
        machines.append({"machine_id": mid, "machine_type": mtype})
        sensors += s
        maint += m
        cycles += c
        sigs += sg
        sig_idx += si
    signals = np.stack(sigs) if sigs else np.empty((0, WINDOW), np.float32)
    sig_index = pd.DataFrame(sig_idx)
    sig_index.insert(0, "signal_idx", np.arange(len(sig_index)))
    return (pd.DataFrame(machines), pd.concat(sensors, ignore_index=True),
            pd.DataFrame(maint), pd.DataFrame(cycles), signals, sig_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-machines", type=int, default=60)
    ap.add_argument("--horizon-h", type=int, default=8000, help="durée d'observation (h)")
    ap.add_argument("--dt-h", type=int, default=6, help="pas entre deux mesures (h)")
    ap.add_argument("--label-h", type=int, default=48, help="'panne dans les H prochaines heures'")
    ap.add_argument("--preventive-h", type=int, default=None,
                    help="intervalle de maintenance préventive fixe (h), vide = aucune")
    ap.add_argument("--signal-every", type=int, default=20,
                    help="1 fenêtre de signal brut toutes les N mesures (0 = pas de signal)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()

    machines, sensors, maint, cycles, signals, sig_index = generate(
        a.n_machines, a.horizon_h, a.dt_h, a.label_h, a.preventive_h, a.seed, a.signal_every)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    machines.to_csv(out / "machines.csv", index=False)
    sensors.round(4).to_csv(out / "sensors.csv", index=False)
    maint.round(2).to_csv(out / "maintenance_log.csv", index=False)
    cycles.round(2).to_csv(out / "survival.csv", index=False)
    np.save(out / "signals.npy", signals)
    sig_index.round(2).to_csv(out / "signals_index.csv", index=False)

    print(f"machines : {len(machines)}")
    print(f"mesures  : {len(sensors)}  (positifs 'panne dans {a.label_h}h' : "
          f"{sensors.failure_within_h.mean():.1%})")
    print(f"cycles   : {len(cycles)}  (pannes observées : {int(cycles.event.sum())}, "
          f"censurés : {int((1 - cycles.event).sum())})")
    print(f"signaux  : {signals.shape[0]} fenêtres de {signals.shape[1]} points @ {FS_HZ} Hz")
    print(f"fichiers écrits dans {out.resolve()}")


if __name__ == "__main__":
    main()
