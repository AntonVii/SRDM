"""
Bernoulli RDM (BRDM) interactive explorer.

Shows the defective (race) PDFs for the winning (correct) and losing
(error) Wald accumulators, plus a panel of simulated diffusion traces
for the same parameters.

Parameterisation:
    pm = Phi( d'/2 - c)      probability the "match" accumulator's spike
                              carries evidence
    pn = Phi(-d'/2 - c)      same for the "nonmatch" accumulator

    v  = p * delta * r        drift
    s  = sqrt(p*(1-p)) * delta * sqrt(r)   diffusion coefficient

    B  = threshold (shared by both accumulators)
    t0 = non-decision time (shift)

Sliders: d', c, delta, B, r, t0
Top panel:    defective RT densities for correct (solid) and error (dashed)
              responses, with accuracy and mean RTs annotated.
Bottom panel: simulated sample paths of the two racing accumulators.

Requires: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ----------------------------------------------------------------------
# Wald (inverse Gaussian) helpers for the race model
# ----------------------------------------------------------------------
def wald_pdf(t, v, s, B):
    """First-passage-time density of a Wald process with drift v,
    diffusion coefficient s, and threshold B."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    ok = t > 0
    out[ok] = (B / np.sqrt(2 * np.pi * s ** 2 * t[ok] ** 3)) * \
              np.exp(-((v * t[ok] - B) ** 2) / (2 * s ** 2 * t[ok]))
    return out


def wald_cdf(t, v, s, B):
    """CDF of the same Wald process."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    ok = t > 0
    ti = t[ok]
    sq = np.sqrt(ti)
    a1 = (v * ti - B) / (s * sq)
    a2 = (v * ti + B) / (s * sq)
    ev = np.clip(2 * B * v / s ** 2, None, 700)
    out[ok] = norm.cdf(a1) + np.exp(ev) * norm.cdf(-a2)
    return np.clip(out, 0, 1)


def sdt_p(dprime, c):
    """SDT mapping: criterion c and discriminability d' -> pm, pn."""
    pm = np.clip(norm.cdf(dprime / 2 - c), 1e-6, 1 - 1e-6)
    pn = np.clip(norm.cdf(-dprime / 2 - c), 1e-6, 1 - 1e-6)
    return pm, pn


def vs_from_p(p, delta, r):
    """Bernoulli-constrained drift and diffusion coefficient."""
    v = p * delta * r
    s = np.sqrt(p * (1 - p)) * delta * np.sqrt(r)
    return v, s


def race_summary(dprime, c, delta, B, r, t0, t_grid):
    """Compute defective densities, accuracy, and mean RTs for the race.

    The "match" accumulator wins on correct trials, "nonmatch" wins on
    error trials. Defective density for correct responses:
        f_correct(t) = f_match(t) * S_nonmatch(t)
    and for errors:
        f_error(t)   = f_nonmatch(t) * S_match(t)
    where S = 1 - CDF (survival function).
    """
    pm, pn = sdt_p(dprime, c)
    vm, sm = vs_from_p(pm, delta, r)
    vn, sn = vs_from_p(pn, delta, r)

    t_eff = np.clip(t_grid - t0, 1e-6, None)

    f_m = wald_pdf(t_eff, vm, sm, B)
    f_n = wald_pdf(t_eff, vn, sn, B)
    S_m = 1 - wald_cdf(t_eff, vm, sm, B)
    S_n = 1 - wald_cdf(t_eff, vn, sn, B)

    f_correct = f_m * S_n
    f_error = f_n * S_m

    # zero out density before t0 (response cannot occur before non-decision time)
    f_correct = np.where(t_grid > t0, f_correct, 0.0)
    f_error = np.where(t_grid > t0, f_error, 0.0)

    # Accuracy and mean RTs via numerical integration over the grid
    dt = t_grid[1] - t_grid[0]
    p_correct = np.sum(f_correct) * dt
    p_error = np.sum(f_error) * dt
    p_total = p_correct + p_error  # should be close to 1

    if p_correct > 1e-8:
        mean_rt_correct = np.sum(t_grid * f_correct) * dt / p_correct
    else:
        mean_rt_correct = np.nan

    if p_error > 1e-8:
        mean_rt_error = np.sum(t_grid * f_error) * dt / p_error
    else:
        mean_rt_error = np.nan

    return dict(
        f_correct=f_correct, f_error=f_error,
        p_correct=p_correct, p_error=p_error, p_total=p_total,
        mean_rt_correct=mean_rt_correct, mean_rt_error=mean_rt_error,
        pm=pm, pn=pn, vm=vm, vn=vn, sm=sm, sn=sn,
    )


# ----------------------------------------------------------------------
# Diffusion simulation for the bottom panel
# ----------------------------------------------------------------------
def simulate_traces(dprime, c, delta, B, r, t0, n_traces=8, t_max=2.0,
                     dt=0.002, seed=0):
    """Simulate sample paths of the two racing accumulators using an
    Euler-Maruyama discretisation of dX = v dt + s dW."""
    rng = np.random.default_rng(seed)
    pm, pn = sdt_p(dprime, c)
    vm, sm = vs_from_p(pm, delta, r)
    vn, sn = vs_from_p(pn, delta, r)

    n_steps = int(t_max / dt)
    times = np.arange(n_steps + 1) * dt

    traces_m = np.zeros((n_traces, n_steps + 1))
    traces_n = np.zeros((n_traces, n_steps + 1))

    sqdt = np.sqrt(dt)
    for i in range(n_traces):
        xm, xn = 0.0, 0.0
        for k in range(1, n_steps + 1):
            xm += vm * dt + sm * sqdt * rng.standard_normal()
            xn += vn * dt + sn * sqdt * rng.standard_normal()
            traces_m[i, k] = xm
            traces_n[i, k] = xn
            # stop both at threshold for cleaner plotting (one race winner)
            if xm >= B or xn >= B:
                traces_m[i, k + 1:] = xm
                traces_n[i, k + 1:] = xn
                break

    return times + t0, traces_m, traces_n, B


# ----------------------------------------------------------------------
# Figure setup
# ----------------------------------------------------------------------
T_MAX = 3.0
N_GRID = 1200
t_grid = np.linspace(1e-4, T_MAX, N_GRID)

# Initial parameter values
init = dict(dprime=1.5, c=0.0, delta=1.0, B=1.0, r=5.0, t0=0.2)

fig, (ax_pdf, ax_sim) = plt.subplots(2, 1, figsize=(9, 9),
                                      gridspec_kw=dict(height_ratios=[1, 1]))
plt.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.40, hspace=0.35)

# --- top panel: defective PDFs -----------------------------------------
line_correct, = ax_pdf.plot([], [], color="#2563eb", lw=2, label="correct")
line_error, = ax_pdf.plot([], [], color="#dc2626", lw=2, ls="--", label="error")
vline_mean_c = ax_pdf.axvline(0, color="#2563eb", lw=1, ls=":", alpha=0.7)
vline_mean_e = ax_pdf.axvline(0, color="#dc2626", lw=1, ls=":", alpha=0.7)

ax_pdf.set_xlim(0, T_MAX)
ax_pdf.set_xlabel("RT (s)")
ax_pdf.set_ylabel("density")
ax_pdf.set_title("Defective RT densities (BRDM Wald race)")
ax_pdf.legend(loc="upper right")
text_stats = ax_pdf.text(0.98, 0.78, "", transform=ax_pdf.transAxes,
                          ha="right", va="top", fontsize=10,
                          family="monospace",
                          bbox=dict(boxstyle="round", fc="white",
                                    ec="0.7", alpha=0.85))

# --- bottom panel: simulated diffusion traces ----------------------------
ax_sim.set_xlim(0, T_MAX)
ax_sim.set_xlabel("time (s)")
ax_sim.set_ylabel("accumulated evidence")
ax_sim.set_title("Simulated accumulator paths (match = blue, nonmatch = red)")
threshold_line = ax_sim.axhline(init["B"], color="0.3", lw=1, ls="--",
                                 label="threshold B")
ax_sim.legend(loc="upper left")
sim_lines_m = []
sim_lines_n = []


# ----------------------------------------------------------------------
# Sliders
# ----------------------------------------------------------------------
slider_axes = {}
sliders = {}
slider_specs = [
    ("dprime", "d'",    0.1, 5.0,  init["dprime"]),
    ("c",      "c",    -2.0, 2.0,  init["c"]),
    ("delta",  "delta", 0.1, 5.0,  init["delta"]),
    ("B",      "B",     0.1, 20.0,  init["B"]),
    ("r",      "r",     0.2, 30.0, init["r"]),
    ("t0",     "t0",    0.0, 0.6,  init["t0"]),
]

slider_top = 0.30
slider_height = 0.03
slider_gap = 0.045

for i, (key, label, vmin, vmax, vinit) in enumerate(slider_specs):
    ax = plt.axes([0.15, slider_top - i * slider_gap, 0.70, slider_height])
    s = Slider(ax, label, vmin, vmax, valinit=vinit)
    slider_axes[key] = ax
    sliders[key] = s


# ----------------------------------------------------------------------
# Update function
# ----------------------------------------------------------------------
def update(_event=None):
    dprime = sliders["dprime"].val
    c = sliders["c"].val
    delta = sliders["delta"].val
    B = sliders["B"].val
    r = sliders["r"].val
    t0 = sliders["t0"].val

    # --- top panel ---
    res = race_summary(dprime, c, delta, B, r, t0, t_grid)
    line_correct.set_data(t_grid, res["f_correct"])
    line_error.set_data(t_grid, res["f_error"])

    ymax = max(res["f_correct"].max(), res["f_error"].max(), 1e-6) * 1.15
    ax_pdf.set_ylim(0, ymax)

    if not np.isnan(res["mean_rt_correct"]):
        vline_mean_c.set_xdata([res["mean_rt_correct"], res["mean_rt_correct"]])
    if not np.isnan(res["mean_rt_error"]):
        vline_mean_e.set_xdata([res["mean_rt_error"], res["mean_rt_error"]])

    stats_str = (
        f"P(correct) = {res['p_correct']:.3f}\n"
        f"P(error)   = {res['p_error']:.3f}\n"
        f"P(total)   = {res['p_total']:.3f}\n"
        f"mean RT correct = {res['mean_rt_correct']:.3f} s\n"
        f"mean RT error   = {res['mean_rt_error']:.3f} s\n"
        f"pm={res['pm']:.3f}  pn={res['pn']:.3f}\n"
        f"vm={res['vm']:.2f}  sm={res['sm']:.2f}\n"
        f"vn={res['vn']:.2f}  sn={res['sn']:.2f}"
    )
    text_stats.set_text(stats_str)

    # --- bottom panel: redraw simulated traces ---
    global sim_lines_m, sim_lines_n
    for ln in sim_lines_m + sim_lines_n:
        ln.remove()
    sim_lines_m, sim_lines_n = [], []

    times, traces_m, traces_n, Bval = simulate_traces(
        dprime, c, delta, B, r, t0, n_traces=6, t_max=T_MAX - t0, seed=1)

    for i in range(traces_m.shape[0]):
        lm, = ax_sim.plot(times, traces_m[i], color="#2563eb", lw=1, alpha=0.6)
        ln, = ax_sim.plot(times, traces_n[i], color="#dc2626", lw=1, alpha=0.6)
        sim_lines_m.append(lm)
        sim_lines_n.append(ln)

    threshold_line.set_ydata([Bval, Bval])
    ax_sim.set_ylim(0, max(Bval * 1.3, traces_m.max(), traces_n.max()) * 1.05)
    ax_sim.set_xlim(0, T_MAX)

    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(update)

update()
plt.show()
