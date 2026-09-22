#!/usr/bin/env python3
"""
Fit DDM and SRDM to rr98 with configurable difficulty resolution.

   ╔══════════════════════════════════════════╗
   ║  CHANGE THIS ONE NUMBER AND RUN AGAIN:  ║
   ║                                         ║
   ║      N_LEVELS = 7   (or 11, 16, 33)     ║
   ╚══════════════════════════════════════════╝

  - 7  = the original qcut binning (fewest params, most trials per bin)
  - 11 = moderate resolution
  - 16 = one level per 2 strength steps
  - 33 = one level per raw strength value (most params, fewest trials per bin)

Uses physical-brightness correctness (strength > 16 = bright).
Excludes strength == 16 (exactly ambiguous).
Fits both DDM and SRDM (c_only) for all 3 participants via optimize().
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# ╔═══════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit these                              ║
# ╚═══════════════════════════════════════════════════════════╝
N_LEVELS     = 33       # <-- CHANGE THIS: 7, 11, 16, 33, etc.
DATA_PATH    = "../rr98.csv"
STAN_DDM     = "DDM_rr98_configurable.stan"
STAN_SRDM    = "SRDM_rr98_c_only_configurable.stan"
PARTICIPANTS = ["jf", "kr", "nh"]


# ═══════════════════════════════════════════════════════════
# Data prep
# ═══════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["outlier"] == False].copy()
    df["correct"] = df["correct"].astype(int)
    df["sat_id"] = df["instruction"].map({"speed": 1, "accuracy": 2})

    # Physical correctness
    df = df[df["strength"] != 16].copy()
    df["act_correct"] = ((df["strength"] > 16) == (df["response"] == "light")).astype(int)

    # Bin strength into N_LEVELS groups
    if N_LEVELS == 33:
        # Special case: use raw strength directly (0-15 -> 1-16, 17-32 -> 17-32)
        # But strength=16 is already excluded, so we have 32 values.
        # Map to 1..32 (not 33) since strength=16 is gone.
        # Actually — qcut with 33 on the remaining 32 unique values will
        # produce 32 bins. Let's just use the raw strength as the level.
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
    print(f"Trials: {len(df)}, cells: {df['cell'].nunique()} (2 x {actual_levels})")
    print(f"Trials per level (min/median/max): "
          f"{df.groupby(['id','cell']).size().min()} / "
          f"{int(df.groupby(['id','cell']).size().median())} / "
          f"{df.groupby(['id','cell']).size().max()}")

    return df, actual_levels


def build_data(df, pid, n_levels):
    d = df[df["id"] == pid]
    d_correct = d[d["act_correct"] == 1]
    d_false = d[d["act_correct"] == 0]
    max_rt = float(d["rt"].max())
    t0_hi = float(d["rt"].quantile(0.05))
    return {
        "N_LEVELS": n_levels,
        "N_correct": len(d_correct), "N_false": len(d_false),
        "rt_correct": d_correct["rt"].to_numpy(),
        "rt_false": d_false["rt"].to_numpy(),
        "cell_correct": d_correct["cell"].to_numpy(dtype=int),
        "cell_false": d_false["cell"].to_numpy(dtype=int),
        "max_rt": max_rt, "t0_hi": t0_hi,
    }


# ═══════════════════════════════════════════════════════════
# Fitting
# ═══════════════════════════════════════════════════════════
def fit_ddm(model, data):
    nl = data["N_LEVELS"]
    inits = {
        "a": [0.8, 1.5], "v_base": [2.0]*nl,
        "sv": 0.5, "sz": 0.1,
        "t0": 0.2 * data["t0_hi"], "p_lapse": 0.02,
    }
    return model.optimize(data=data, inits=inits, algorithm="lbfgs",
                          iter=5000, show_console=True)


def fit_srdm(model, data):
    nl = data["N_LEVELS"]
    inits = {
        "c": [0.0, 0.0], "B": 1.0,
        "t0": 0.5 * data["t0_hi"],
        "d_base": [1.5]*nl, "r": 4.5, "p_lapse": 0.02,
    }
    return model.optimize(data=data, inits=inits, algorithm="lbfgs",
                          iter=5000, show_console=True)


def aic_bic(mle, n_params):
    p = mle.optimized_params_pd
    ll_cols = [c for c in p.columns if c.startswith("log_lik")]
    total_ll = p[ll_cols].iloc[0].sum()
    n = len(ll_cols)
    return {"n_params": n_params, "n_trials": n, "log_lik": total_ll,
            "AIC": 2*n_params - 2*total_ll,
            "BIC": n_params*np.log(n) - 2*total_ll}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    df, actual_levels = load_data()

    # Parameter counts: a[2] + v_or_d[actual_levels] + sv + sz + t0 + p_lapse
    ddm_n_params  = 2 + actual_levels + 4   # a, v_base, sv, sz, t0, p_lapse
    srdm_n_params = 2 + actual_levels + 3   # c, d_base, B, r, t0, p_lapse

    print(f"\nDDM params: {ddm_n_params} ({actual_levels} drift rates)")
    print(f"SRDM params: {srdm_n_params} ({actual_levels} d' values)")

    print("\nCompiling models...")
    ddm_model = CmdStanModel(stan_file=STAN_DDM)
    srdm_model = CmdStanModel(stan_file=STAN_SRDM)

    for pid in PARTICIPANTS:
        print(f"\n{'='*60}")
        print(f"  {pid}  (N_LEVELS={actual_levels})")
        print(f"{'='*60}")

        data = build_data(df, pid, actual_levels)

        # DDM
        print(f"\n  --- DDM ---")
        ddm_mle = fit_ddm(ddm_model, data)
        fname = f"map_{actual_levels}lev_ddm_{pid}.csv"
        ddm_mle.optimized_params_pd.to_csv(fname, index=False)
        ic = aic_bic(ddm_mle, ddm_n_params)
        row = ddm_mle.optimized_params_pd.iloc[0]
        print(f"  a=[{row['a[1]']:.3f}, {row['a[2]']:.3f}]  t0={row['t0']:.4f}  "
              f"sv={row['sv']:.4f}  sz={row['sz']:.4f}  p_lapse={row['p_lapse']:.4f}")
        print(f"  LL={ic['log_lik']:.1f}  AIC={ic['AIC']:.1f}  BIC={ic['BIC']:.1f}")

        # SRDM
        print(f"\n  --- SRDM ---")
        srdm_mle = fit_srdm(srdm_model, data)
        fname = f"map_{actual_levels}lev_srdm_{pid}.csv"
        srdm_mle.optimized_params_pd.to_csv(fname, index=False)
        ic2 = aic_bic(srdm_mle, srdm_n_params)
        row = srdm_mle.optimized_params_pd.iloc[0]
        print(f"  c=[{row['c[1]']:.3f}, {row['c[2]']:.3f}]  B={row['B']:.3f}  "
              f"r={row['r']:.2f}  t0={row['t0']:.4f}  p_lapse={row['p_lapse']:.4f}")
        print(f"  LL={ic2['log_lik']:.1f}  AIC={ic2['AIC']:.1f}  BIC={ic2['BIC']:.1f}")

        delta = ic["AIC"] - ic2["AIC"]
        print(f"\n  ΔAIC(DDM-SRDM) = {delta:+.1f}  ({'DDM' if delta<0 else 'SRDM'} wins)")


if __name__ == "__main__":
    main()
