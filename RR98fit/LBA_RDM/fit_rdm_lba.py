#!/usr/bin/env python3
"""
Fit RDM (shared/varying v_error) and LBA (shared/varying v_error) to rr98,
matching the exact same no-lapse, configurable-binning, trimmed-data
convention as the DDM/SRDM fitting scripts elsewhere in this project.

OUTPUT FILES (one per model variant, all participants + N_LEVELS stacked):
  - fits_rdm_shared_verr.csv
  - fits_rdm_varying_verr.csv
  - fits_lba_shared_verr.csv
  - fits_lba_varying_verr.csv
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# ╔═══════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these                              ║
# ╚═══════════════════════════════════════════════════════════╝
N_LEVELS      = 16      # <-- CHANGE THIS: 7, 11, 16, 33, etc. -- same
                        #     convention as the DDM/SRDM scripts. Loop
                        #     over a list of values the same way you did
                        #     for the DDM/SRDM resolution sweep.
PARTICIPANTS  = ["jf", "kr", "nh"]
TRIM_LOW      = 0.01
TRIM_HIGH     = 0.99

MODELS = {
    "rdm_shared":  ("RDM_shared_verr.stan",  "fits_rdm_shared_verr.csv"),
    "rdm_varying": ("RDM_varying_verr.stan", "fits_rdm_varying_verr.csv"),
    "lba_shared":  ("LBA_shared_verr.stan",  "fits_lba_shared_verr.csv"),
    "lba_varying": ("LBA_varying_verr.stan", "fits_lba_varying_verr.csv"),
}




def load_data():
    df = pd.read_csv("../rr98.csv")
    df = df[df["outlier"] == False].copy()
    df["sat_id"] = df["instruction"].map({"speed": 1, "accuracy": 2})
    df = df[df["strength"] != 16].copy()
    df["act_correct"] = ((df["strength"] > 16) == (df["response"] == "light")).astype(int)

    if N_LEVELS == 33:
        unique_strengths = sorted(df["strength"].unique())
        strength_to_level = {s: i+1 for i, s in enumerate(unique_strengths)}
        df["diff_level"] = df["strength"].map(strength_to_level)
        actual_levels = len(unique_strengths)
    else:
        def _qcut_levels(s):
            return pd.qcut(s, q=N_LEVELS, labels=False, duplicates="drop") + 1
        df["diff_level"] = df.groupby("id")["strength"].transform(_qcut_levels)
        actual_levels = df["diff_level"].nunique()

    df["cell"] = (df["sat_id"] - 1) * actual_levels + df["diff_level"]
    print(f"N_LEVELS requested: {N_LEVELS}, actual unique levels: {actual_levels}")
    return df, actual_levels


def trim_extremes(df, low=TRIM_LOW, high=TRIM_HIGH):
    def _trim_group(g):
        lo, hi = g["rt"].quantile([low, high])
        return g[(g["rt"] >= lo) & (g["rt"] <= hi)]
    return df.groupby(["id", "cell"], group_keys=False).apply(_trim_group)


def build_data(df, pid, n_levels):
    d = df[df["id"] == pid]
    d_correct = d[d["act_correct"] == 1]
    d_false = d[d["act_correct"] == 0]
    t0_hi = float(d["rt"].min())
    return {
        "N_LEVELS": n_levels,
        "N_correct": len(d_correct), "N_false": len(d_false),
        "rt_correct": d_correct["rt"].to_numpy(),
        "rt_false": d_false["rt"].to_numpy(),
        "cell_correct": d_correct["cell"].to_numpy(dtype=int),
        "cell_false": d_false["cell"].to_numpy(dtype=int),
        "t0_hi": t0_hi,
    }


def get_inits(model_key, n_levels, t0_hi):
    if model_key.startswith("rdm"):
        base = {"B": [1.0, 1.5], "A": 0.4, "v_correct": [2.5]*n_levels, "t0": 0.2*t0_hi}
    else:  # lba
        base = {"k": [0.6, 0.9], "A": 0.4, "v_correct": [2.5]*n_levels, "t0": 0.2*t0_hi}
    if model_key.endswith("shared"):
        base["v_error"] = 1.0
    else:
        base["v_error"] = [1.0]*n_levels
    return base


def n_params_for(model_key, n_levels):
    # B/k[2] + A + v_correct[N] + v_error(1 or N) + t0
    v_err_params = 1 if model_key.endswith("shared") else n_levels
    return 2 + 1 + n_levels + v_err_params + 1


def aic_bic(mle, n_params):
    p = mle.optimized_params_pd
    ll_cols = [c for c in p.columns if c.startswith("log_lik")]
    total_ll = p[ll_cols].iloc[0].sum()
    n = len(ll_cols)
    return {"n_params": n_params, "n_trials": n, "log_lik": total_ll,
            "AIC": 2*n_params - 2*total_ll, "BIC": n_params*np.log(n) - 2*total_ll}


def save_row(csv_path, pid, n_levels_requested, n_levels_actual, mle, ic, t0_hi):
    raw = mle.optimized_params_pd.iloc[0]
    keep_cols = [c for c in raw.index if not c.startswith("log_lik")]
    row = raw[keep_cols].to_dict()
    row = {"pid": pid, "n_levels_requested": n_levels_requested,
           "n_levels_actual": n_levels_actual, "n_params": ic["n_params"],
           "n_trials": ic["n_trials"], "log_lik_total": ic["log_lik"],
           "AIC": ic["AIC"], "BIC": ic["BIC"], "t0_hi": t0_hi,
           "fit_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **row}
    new_row = pd.DataFrame([row])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        mask = (existing["pid"] == pid) & (existing["n_levels_actual"] == n_levels_actual)
        existing = existing[~mask]
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined = combined.sort_values(["n_levels_actual", "pid"]).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)


def main():
    df, actual_levels = load_data()
    df = trim_extremes(df)

    compiled = {}
    for model_key, (stan_file, _) in MODELS.items():
        print(f"Compiling {model_key} ({stan_file})...")
        compiled[model_key] = CmdStanModel(stan_file=stan_file)

    for pid in PARTICIPANTS:
        print(f"\n{'='*60}\n  {pid}  (N_LEVELS={actual_levels})\n{'='*60}")
        data = build_data(df, pid, actual_levels)
        print(f"  N_correct={data['N_correct']}  N_false={data['N_false']}  "
              f"t0_hi={data['t0_hi']:.4f}")

        for model_key, (stan_file, out_path) in MODELS.items():
            print(f"\n  --- {model_key} ---")
            inits = get_inits(model_key, actual_levels, data["t0_hi"])
            mle = compiled[model_key].optimize(data=data, inits=inits, algorithm="lbfgs",
                                               iter=5000, show_console=False)
            n_params = n_params_for(model_key, actual_levels)
            ic = aic_bic(mle, n_params)
            save_row(out_path, pid, N_LEVELS, actual_levels, mle, ic, data["t0_hi"])
            print(f"  n_params={n_params}  LL={ic['log_lik']:.1f}  "
                  f"AIC={ic['AIC']:.1f}  BIC={ic['BIC']:.1f}")

    print("\nSaved:")
    for _, out_path in MODELS.values():
        print(f"  {out_path}")


if __name__ == "__main__":
    main()
