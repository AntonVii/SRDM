#!/usr/bin/env python3
"""
Recreate the rtdists (Singmann et al.) reanalysis scheme for RR98 using
our own Stan/Blurton-density pipeline: same 5 strength bins, same
per-participant-x-instruction separate fits, raw response coding (not
correctness-recoded). If we reproduce their qualitative pattern (sv -> 0
in speed, sz speed > accuracy) with completely different code, that's
strong confirmation the result is about the data/model, not our pipeline.

OUTPUT: fits_reanalysis_style.csv, one row per (pid, instruction).
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

STAN_FILE = "DDM_rr98_reanalysis_style.stan"
PARTICIPANTS = ["jf", "kr", "nh"]
INSTRUCTIONS = ["speed", "accuracy"]
TRIM_LOW, TRIM_HIGH = 0.01, 0.99

# Exact bin edges from the rtdists vignette (5 bins)
BIN_EDGES = [-0.5, 10.5, 13.5, 16.5, 19.5, 32.5]
N_LEVELS = len(BIN_EDGES) - 1

OUT_PATH = "fits_reanalysis_style.csv"


def find_rr98_csv(filename="rr98.csv", max_up=6):
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents][:max_up + 1]:
        candidate = parent / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Could not find {filename} near {here}")


def load_data():
    df = pd.read_csv(find_rr98_csv())
    df = df[df["outlier"] == False].copy()
    df["level"] = pd.cut(df["strength"], bins=BIN_EDGES, labels=False) + 1  # 1-indexed
    return df


def trim_extremes(df, low=TRIM_LOW, high=TRIM_HIGH):
    """Trim fastest/slowest tails within each (id, instruction, level, response)
    group -- as fine-grained as the earlier no-lapse trimming, just now also
    split by response identity since that's the unit of analysis here."""
    def _trim_group(g):
        lo, hi = g["rt"].quantile([low, high])
        return g[(g["rt"] >= lo) & (g["rt"] <= hi)]
    return df.groupby(["id", "instruction", "level"], group_keys=False).apply(_trim_group)


def build_data(df, pid, instruction):
    d = df[(df["id"] == pid) & (df["instruction"] == instruction)]
    d_dark = d[d["response"] == "dark"]
    d_light = d[d["response"] == "light"]
    t0_hi = float(d["rt"].min())
    return {
        "N_LEVELS": N_LEVELS,
        "N_dark": len(d_dark), "N_light": len(d_light),
        "rt_dark": d_dark["rt"].to_numpy(), "rt_light": d_light["rt"].to_numpy(),
        "level_dark": d_dark["level"].to_numpy(dtype=int),
        "level_light": d_light["level"].to_numpy(dtype=int),
        "t0_hi": t0_hi,
    }


def fit(model, data):
    inits = {"a": 1.5, "v": [0.0]*N_LEVELS, "z_rel": 0.5, "sv": 0.5, "sz": 0.1,
             "t0": 0.2 * data["t0_hi"]}
    return model.optimize(data=data, inits=inits, algorithm="lbfgs",
                          iter=5000, show_console=False)


def aic_bic(mle, n_params):
    p = mle.optimized_params_pd
    ll_cols = [c for c in p.columns if c.startswith("log_lik")]
    total_ll = p[ll_cols].iloc[0].sum()
    n = len(ll_cols)
    return {"n_params": n_params, "n_trials": n, "log_lik": total_ll,
            "AIC": 2*n_params - 2*total_ll, "BIC": n_params*np.log(n) - 2*total_ll}


def save_row(csv_path, pid, instruction, mle, ic, t0_hi):
    raw = mle.optimized_params_pd.iloc[0]
    keep_cols = [c for c in raw.index if not c.startswith("log_lik")]
    row = raw[keep_cols].to_dict()
    row = {"pid": pid, "instruction": instruction, "n_params": ic["n_params"],
           "n_trials": ic["n_trials"], "log_lik_total": ic["log_lik"],
           "AIC": ic["AIC"], "BIC": ic["BIC"], "t0_hi": t0_hi,
           "fit_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **row}
    new_row = pd.DataFrame([row])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        mask = (existing["pid"] == pid) & (existing["instruction"] == instruction)
        existing = existing[~mask]
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined = combined.sort_values(["pid", "instruction"]).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)


def main():
    df = load_data()
    df = trim_extremes(df)

    n_params = 1 + N_LEVELS + 1 + 1 + 1 + 1  # a, v[5], z_rel, sv, sz, t0
    print(f"Params per fit: {n_params} ({N_LEVELS} drift rates + a, z_rel, sv, sz, t0)")

    print("Compiling model...")
    model = CmdStanModel(stan_file=STAN_FILE)

    for pid in PARTICIPANTS:
        for instruction in INSTRUCTIONS:
            print(f"\n=== {pid} / {instruction} ===")
            data = build_data(df, pid, instruction)
            print(f"  N_dark={data['N_dark']}  N_light={data['N_light']}  "
                  f"t0_hi={data['t0_hi']:.4f}")
            mle = fit(model, data)
            ic = aic_bic(mle, n_params)
            save_row(OUT_PATH, pid, instruction, mle, ic, data["t0_hi"])
            row = mle.optimized_params_pd.iloc[0]
            print(f"  a={row['a']:.3f}  z_rel={row['z_rel']:.3f}  t0={row['t0']:.4f}  "
                  f"sv={row['sv']:.4f}  sz={row['sz']:.4f}")
            print(f"  LL={ic['log_lik']:.1f}  AIC={ic['AIC']:.1f}  BIC={ic['BIC']:.1f}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
