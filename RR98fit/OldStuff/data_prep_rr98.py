import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------

df = pd.read_csv("rr98.csv")

# drop outlier trials (package docs: RT < 180ms or > 3s, or uninterpretable response)
df = df[df["outlier"] == False].copy()

df["correct"] = df["correct"].astype(int)
df["sat_id"] = df["instruction"].map({"speed": 1, "accuracy": 2})
assert df["sat_id"].notna().all(), "unmapped instruction labels found"

# ---------------------------------------------------------------------
# 2. Collapse 'strength' (raw ~1-33 scale) into 7 difficulty levels, PER
#    PARTICIPANT (quantile-based, so each level has a comparable trial
#    count and uses that participant's own observed range - only 3
#    participants here, no pooling across them, so this is the natural
#    choice rather than a single global binning).
# ---------------------------------------------------------------------

def _qcut_levels(s, n_levels=7):
    return pd.qcut(s, q=n_levels, labels=False, duplicates="drop") + 1

df["diff_level"] = df.groupby("id")["strength"].transform(_qcut_levels)
assert df["diff_level"].notna().all(), "difficulty binning failed for some rows"

# combined SAT x difficulty cell index: 1..14
# cell = (sat_id - 1) * 7 + diff_level
df["cell"] = (df["sat_id"] - 1) * 7 + df["diff_level"]


# ---------------------------------------------------------------------
# 3. Per-participant, per-model-variant data dict builder
# ---------------------------------------------------------------------

VARIANT_T0_VARIES = {
    "c_B": False, "c_t0": True, "d_B": False, "d_t0": True,
}

def build_data(participant_id, variant):
    d = df[df["id"] == participant_id]
    d_correct = d[d["correct"] == 1]
    d_false = d[d["correct"] == 0]

    max_rt = float(d["rt"].max())  # pooled lapse range, single scalar

    t0_varies = VARIANT_T0_VARIES[variant]
    if t0_varies:
        t0_hi = d.groupby("sat_id")["rt"].quantile(0.05).reindex([1, 2])
        assert t0_hi.notna().all(), f"empty SAT cell for participant {participant_id}"
        t0_hi_val = t0_hi.to_numpy()
    else:
        t0_hi_val = float(d["rt"].quantile(0.05))

    data = {
        "N_correct": len(d_correct),
        "N_false": len(d_false),
        "rt_correct": d_correct["rt"].to_numpy(),
        "rt_false": d_false["rt"].to_numpy(),
        "cell_correct": d_correct["cell"].to_numpy(dtype=int),
        "cell_false": d_false["cell"].to_numpy(dtype=int),
        "max_rt": max_rt,
        "t0_hi": t0_hi_val,
    }
    return data


# ---------------------------------------------------------------------
# 4. Init values, matched to each variant's parameter shapes
# ---------------------------------------------------------------------

def init_c_B():
    return {
        "c": [0.0, 0.0], "B": [1.0, 1.0],
        "t0": 0.2, "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02,
    }

def init_c_t0():
    return {
        "c": [0.0, 0.0], "t0": [0.2, 0.2],
        "B": 1.0, "d_base": [1.5]*7, "r": 4.5, "p_lapse": 0.02,
    }

def init_d_B():
    return {
        "c": 0.0, "B": [1.0, 1.0], "t0": 0.2,
        "d_base": [1.5]*7, "delta_d": 0.0, "r": 4.5, "p_lapse": 0.02,
    }

def init_d_t0():
    return {
        "c": 0.0, "B": 1.0, "t0": [0.2, 0.2],
        "d_base": [1.5]*7, "delta_d": 0.0, "r": 4.5, "p_lapse": 0.02,
    }

INIT_FNS = {"c_B": init_c_B, "c_t0": init_c_t0, "d_B": init_d_B, "d_t0": init_d_t0}


# ---------------------------------------------------------------------
# Example usage:
#
# from cmdstanpy import CmdStanModel
# model = CmdStanModel(stan_file="SRDM_rr98_c_B.stan")
# data = build_data("jf", "c_B")
# fit = model.sample(data=data, chains=4, inits=INIT_FNS["c_B"](),
#                     iter_warmup=1000, iter_sampling=1000, adapt_delta=0.95)
# ---------------------------------------------------------------------
