#!/usr/bin/env python3
"""
Profile likelihood over a grid of FIXED sv values for the no-lapse DDM.

For each participant and each fixed sv in the grid, refits everything
else (a, v_base, sz, t0) via MAP, and records the total log-likelihood.
This answers directly: does the likelihood, with everything else allowed
to re-optimize, actually keep improving as sv -> 0 (genuine, persistent
pull toward the boundary), or is there an interior maximum somewhere
that the joint optimizer has been missing?

Run this on your machine (needs cmdstanpy + a compiled CmdStan + rr98.csv).

OUTPUT: profile_sv.csv, one row per (pid, sv_fixed).
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

N_LEVELS = 16
STAN_DDM = "DDM_rr98_nolapse_profile_sv.stan"
PARTICIPANTS = ["jf", "kr", "nh"]
TRIM_LOW, TRIM_HIGH = 0.01, 0.99


def find_rr98_csv(filename="rr98.csv", max_up=6):
    """Search the script's own directory and its parents for rr98.csv,
    instead of assuming a fixed '../../rr98.csv' relative path -- that
    path depends on exactly which subfolder the script happens to sit
    in, which is fragile the moment you move it (e.g. into a new
    SvSzbyCond/ subfolder one level deeper than before)."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents][:max_up + 1]:
        candidate = parent / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find {filename} in {here} or any of its parents "
        f"(searched {max_up} levels up). Either move this script next "
        f"to where your other fitting scripts live, or hardcode "
        f"DATA_PATH below to the correct path."
    )


DATA_PATH = find_rr98_csv()
print(f"Using data file: {DATA_PATH}")

# The grid to profile over. Dense near 0 (where the action is), sparser
# further out. Add/remove points as you like -- denser = smoother curve,
# more compute.
SV_GRID = np.concatenate([
    [1e-4, 0.01, 0.02, 0.03, 0.05],
    np.arange(0.1, 1.01, 0.1),
])

OUT_PATH = "profile_sv.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    required_cols = {"id", "outlier", "instruction", "strength", "response", "rt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Loaded {DATA_PATH}, but it's missing expected column(s): {missing}. "
            f"Columns found: {list(df.columns)}. This usually means DATA_PATH "
            f"resolved to the wrong file -- check find_rr98_csv() above."
        )
    df = df[df["outlier"] == False].copy()
    df["sat_id"] = df["instruction"].map({"speed": 1, "accuracy": 2})
    df = df[df["strength"] != 16].copy()
    df["act_correct"] = ((df["strength"] > 16) == (df["response"] == "light")).astype(int)

    def _qcut_levels(s):
        return pd.qcut(s, q=N_LEVELS, labels=False, duplicates="drop") + 1
    df["diff_level"] = df.groupby("id")["strength"].transform(_qcut_levels)
    actual_levels = df["diff_level"].nunique()
    df["cell"] = (df["sat_id"] - 1) * actual_levels + df["diff_level"]
    return df, actual_levels


def trim_extremes(df, low=TRIM_LOW, high=TRIM_HIGH):
    def _trim_group(g):
        lo, hi = g["rt"].quantile([low, high])
        return g[(g["rt"] >= lo) & (g["rt"] <= hi)]
    return df.groupby(["id", "cell"], group_keys=False).apply(_trim_group)


def build_data(df, pid, n_levels, sv_fixed):
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
        "sv_fixed": float(sv_fixed),
    }


def fit_at_fixed_sv(model, data):
    nl = data["N_LEVELS"]
    inits = {"a": [0.8, 1.5], "v_base": [2.0]*nl, "sz": 0.1,
             "t0": 0.2 * data["t0_hi"]}
    return model.optimize(data=data, inits=inits, algorithm="lbfgs",
                          iter=5000, show_console=False)


def total_log_lik(mle):
    p = mle.optimized_params_pd
    ll_cols = [c for c in p.columns if c.startswith("log_lik")]
    return p[ll_cols].iloc[0].sum()


def main():
    df, actual_levels = load_data()
    df = trim_extremes(df)
    print(f"Using {actual_levels} difficulty levels, {len(SV_GRID)} sv values, "
          f"{len(PARTICIPANTS)} participants -> {len(SV_GRID)*len(PARTICIPANTS)} fits total.")

    print("Compiling model...")
    model = CmdStanModel(stan_file=STAN_DDM)

    rows = []
    for pid in PARTICIPANTS:
        print(f"\n=== {pid} ===")
        for sv_fixed in SV_GRID:
            data = build_data(df, pid, actual_levels, sv_fixed)
            try:
                mle = fit_at_fixed_sv(model, data)
                ll = total_log_lik(mle)
                row = mle.optimized_params_pd.iloc[0]
                print(f"  sv={sv_fixed:.4f}  log_lik={ll:.2f}  "
                      f"sz={row['sz']:.4f}  t0={row['t0']:.4f}")
                rows.append({"pid": pid, "sv_fixed": sv_fixed, "log_lik": ll,
                             "sz": row["sz"], "t0": row["t0"],
                             "a1": row["a[1]"], "a2": row["a[2]"]})
            except Exception as e:
                print(f"  sv={sv_fixed:.4f}  FAILED: {e}")
                rows.append({"pid": pid, "sv_fixed": sv_fixed, "log_lik": np.nan,
                             "sz": np.nan, "t0": np.nan, "a1": np.nan, "a2": np.nan})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")

    # Quick summary: where does the profile peak for each participant?
    for pid in PARTICIPANTS:
        sub = out[out["pid"] == pid].dropna()
        best = sub.loc[sub["log_lik"].idxmax()]
        print(f"{pid}: profile-likelihood peak at sv={best['sv_fixed']:.4f} "
              f"(log_lik={best['log_lik']:.2f})")


if __name__ == "__main__":
    main()
