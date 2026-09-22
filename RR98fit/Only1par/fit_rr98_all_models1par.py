"""
Fit all four rr98 model variants (c/B, c/t0, d'/B, d'/t0) to all 3
participants, in parallel.

Saves, per (participant, variant) fit:
  1. per_fit_summaries.csv   - parameter estimates, with labels, tagged
  2. diagnostics.csv         - divergences, treedepth, max Rhat, min ESS
  3. idata/{variant}_{pid}.nc - full posterior + log_lik, for LOO
"""

import os
import time
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # avoid the file-lock crash from before

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel
from joblib import Parallel, delayed


# ---------------------------------------------------------------------
# 1. Load and prep data (same as data_prep_rr98.py)
# ---------------------------------------------------------------------

df = pd.read_csv("../rr98.csv")
df = df[df["outlier"] == False].copy()
df["correct"] = df["correct"].astype(int)
df["sat_id"] = df["instruction"].map({"speed": 1, "accuracy": 2})
assert df["sat_id"].notna().all(), "unmapped instruction labels found"

def _qcut_levels(s, n_levels=7):
    return pd.qcut(s, q=n_levels, labels=False, duplicates="drop") + 1

df["diff_level"] = df.groupby("id")["strength"].transform(_qcut_levels)
assert df["diff_level"].notna().all(), "difficulty binning failed for some rows"

df["cell"] = (df["sat_id"] - 1) * 7 + df["diff_level"]

PARTICIPANTS = sorted(df["id"].unique())
print(f"Participants found: {PARTICIPANTS}")
for pid in PARTICIPANTS:
    n_levels = df[df["id"] == pid]["diff_level"].nunique()
    n_trials = len(df[df["id"] == pid])
    print(f"  {pid}: {n_trials} trials, {n_levels} difficulty levels")


# ---------------------------------------------------------------------
# 2. Variant specs: stan file, init function, whether t0 varies by SAT
# ---------------------------------------------------------------------

def init_c_B(data):
    return {"c": [0.0, 0.0], "B": [1.0, 1.0],
            "t0": 0.5 * data["t0_hi"],  # safe fraction of this participant's actual bound
            "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02}

def init_c_t0(data):
    return {"c": [0.0, 0.0], "t0": [0.5 * v for v in data["t0_hi"]],
            "B": 1.0, "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02}

def init_d_B(data):
    return {"c": 0.0, "B": [1.0, 1.0],
            "t0": 0.5 * data["t0_hi"],
            "d_base": [1.5]*7, "delta_d": 0.0, "r": 4.5, "p_lapse": 0.02}

def init_d_t0(data):
    return {"c": 0.0, "B": 1.0, "t0": [0.5 * v for v in data["t0_hi"]],
            "d_base": [1.5]*7, "delta_d": 0.0, "r": 4.5, "p_lapse": 0.02}

# ---------------------------------------------------------------------
# 2b. Per-participant data builder (defined before model compilation
#     since the warm-up call below needs it)
# ---------------------------------------------------------------------

def build_data(participant_id, t0_varies):
    d = df[df["id"] == participant_id]
    d_correct = d[d["correct"] == 1]
    d_false = d[d["correct"] == 0]

    max_rt = float(d["rt"].max())

    if t0_varies:
        t0_hi = d.groupby("sat_id")["rt"].quantile(0.05).reindex([1, 2])
        assert t0_hi.notna().all(), f"empty SAT cell for participant {participant_id}"
        t0_hi_val = t0_hi.to_numpy()
    else:
        t0_hi_val = float(d["rt"].quantile(0.05))

    return {
        "N_correct": len(d_correct),
        "N_false": len(d_false),
        "rt_correct": d_correct["rt"].to_numpy(),
        "rt_false": d_false["rt"].to_numpy(),
        "cell_correct": d_correct["cell"].to_numpy(dtype=int),
        "cell_false": d_false["cell"].to_numpy(dtype=int),
        "max_rt": max_rt,
        "t0_hi": t0_hi_val,
    }


def init_c_only(data):
    return {"c": [0.0, 0.0], "B": 1.0, "t0": 0.5 * data["t0_hi"],
            "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02}

def init_B_only(data):
    return {"c": 0.0, "B": [1.0, 1.0], "t0": 0.5 * data["t0_hi"],
            "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02}

def init_t0_only(data):
    return {"c": 0.0, "B": 1.0, "t0": [0.5 * v for v in data["t0_hi"]],
            "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02}

def init_d_only(data):
    return {"c": 0.0, "B": 1.0, "t0": 0.5 * data["t0_hi"],
            "d_base": [1.5]*7, "delta_d": 0.0, "r": 4.5, "p_lapse": 0.02}


MODEL_SPECS = [
    {"name": "c_only",  "stan_file": "SRDM_rr98_c_only.stan",  "init_fn": init_c_only,  "t0_varies": False},
    {"name": "B_only",  "stan_file": "SRDM_rr98_B_only.stan",  "init_fn": init_B_only,  "t0_varies": False},
    {"name": "t0_only", "stan_file": "SRDM_rr98_t0_only.stan", "init_fn": init_t0_only, "t0_varies": True},
    {"name": "d_only",  "stan_file": "SRDM_rr98_d_only.stan",  "init_fn": init_d_only,  "t0_varies": False},
]

for spec in MODEL_SPECS:
    spec["model"] = CmdStanModel(stan_file=spec["stan_file"])

    # Retry the warm-up call a few times with backoff: freshly-created files
    # in an iCloud-synced folder (this is a brand-new project directory,
    # unlike the older, already-settled Forstmann folder) can transiently
    # block execution while the sync daemon is still touching them.
    _warmup_data = build_data(PARTICIPANTS[0], spec["t0_varies"])
    for attempt in range(5):
        try:
            spec["model"].sample(
                data=_warmup_data, chains=1, iter_warmup=10, iter_sampling=10,
                inits=spec["init_fn"](_warmup_data), show_progress=False, show_console=False,
            )
            print(f"  warmed up {spec['name']} (attempt {attempt + 1})")
            break
        except RuntimeError as e:
            wait = 5 * (attempt + 1)
            print(f"  warm-up failed for {spec['name']} (attempt {attempt + 1}), "
                  f"retrying in {wait}s: {e}")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Could not warm up {spec['name']} after 5 attempts")


# ---------------------------------------------------------------------
# 3. (build_data defined above, in section 2b)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 4. Fit one (participant, variant) pair
# ---------------------------------------------------------------------

def fit_one(pid, spec, idata_dir):
    try:
        data = build_data(pid, spec["t0_varies"])
        init_values = spec["init_fn"](data)

        fit = spec["model"].sample(
            data=data, chains=4, parallel_chains=1,
            inits=init_values, iter_warmup=1000, iter_sampling=1000,
            adapt_delta=0.95, show_progress=False,
        )

        summary = fit.summary().reset_index().rename(columns={"index": "param"})
        summary["participant_id"] = pid
        summary["model"] = spec["name"]

        sv = fit.method_variables()
        n_iter_total = sv["divergent__"].size
        diagnostics = {
            "participant_id": pid,
            "model": spec["name"],
            "n_divergent": int(sv["divergent__"].sum()),
            "pct_divergent": float(sv["divergent__"].sum()) / n_iter_total,
            "mean_treedepth": float(np.mean(sv["treedepth__"])),
            "max_treedepth": float(np.max(sv["treedepth__"])),
            "mean_n_leapfrog": float(np.mean(sv["n_leapfrog__"])),
            "max_rhat": float(summary["R_hat"].max()),
            "min_ess_bulk": float(summary["ESS_bulk"].min()),
        }

        try:
            idata = az.from_cmdstanpy(fit, log_likelihood="log_lik")
            idata.to_netcdf(idata_dir / f"{spec['name']}_{pid}.nc")
        except Exception as save_exc:
            print(f"[SAVE FAILED, keeping other results] participant={pid} "
                  f"model={spec['name']}: {save_exc}")

        return summary, diagnostics

    except Exception as exc:
        print(f"[FAILED] participant={pid} model={spec['name']}: {exc}")
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------
# 5. Run: outer loop over models, inner parallel loop over participants
# ---------------------------------------------------------------------

def run_all(out_dir="../rr98_fits"):
    out_dir = Path(out_dir)
    idata_dir = out_dir / "idata"
    out_dir.mkdir(parents=True, exist_ok=True)
    idata_dir.mkdir(parents=True, exist_ok=True)

    all_summaries, all_diagnostics = [], []

    for spec in MODEL_SPECS:
        print(f"=== Fitting model: {spec['name']} ({len(PARTICIPANTS)} participants) ===")

        results = Parallel(n_jobs=-1)(
            delayed(fit_one)(pid, spec, idata_dir) for pid in PARTICIPANTS
        )
        summaries = [s for s, _ in results if s is not None]
        diags = [d for _, d in results if d is not None]

        if summaries:
            model_df = pd.concat(summaries)
            model_df.to_csv(out_dir / f"per_subject_fits_{spec['name']}.csv", index=False)
            all_summaries.append(model_df)

        if diags:
            diag_df = pd.DataFrame(diags)
            diag_df.to_csv(out_dir / f"diagnostics_{spec['name']}.csv", index=False)
            all_diagnostics.append(diag_df)

            n_bad = (diag_df["pct_divergent"] > 0.01).sum()
            print(f"  -> {n_bad}/{len(diag_df)} participants had >1% divergent transitions")
        else:
            print(f"  !! no successful fits for model {spec['name']}")

    combined_summary = pd.concat(all_summaries) if all_summaries else pd.DataFrame()
    combined_diag = pd.concat(all_diagnostics) if all_diagnostics else pd.DataFrame()

    combined_summary.to_csv(out_dir / "per_subject_fits_all_models.csv", index=False)
    combined_diag.to_csv(out_dir / "diagnostics_all_models.csv", index=False)

    return combined_summary, combined_diag


if __name__ == "__main__":
    all_results, all_diagnostics = run_all(out_dir="rr98_fits")
