// RDM (Racing Diffusion Model, Tillman, Van Zandt & Logan 2020) for rr98.
// N_LEVELS is passed in as data. Given the parameter count below, this is
// meant to be run at SMALL N_LEVELS (5-7), not 16+.
//
// PURPOSE: B is fixed externally (set to SRDM's own fitted boundary
// value) to anchor scale, then v_correct, v_error, s_correct, s_error are
// all freely estimated varying by BOTH difficulty level AND SAT
// condition (speed/accuracy) -- the goal being to see how close the
// freely-estimated structure comes to the SRDM's own d'/c-based
// parameter reduction.
//
// Total free params: 4 (v_correct, v_error, s_correct, s_error) x N_LEVELS
// x 2 conditions + A + t0 = 8*N_LEVELS + 2. At N_LEVELS=7 that's 58; keep
// N_LEVELS small (5-7) given typical per-cell trial counts.
//
// FIXES vs. the previous version:
//   1. rdm_pdf's noise scaling was wrong for general s -- it had been
//      copied from the LBA's ballistic-process formula (which scales as
//      t*s, appropriate for a straight-line accumulation process) rather
//      than the correct diffusion scaling (s*sqrt(t), since a Wiener
//      process's variance grows with t, not t^2). Fixed and verified
//      against an independent unit-noise substitution (a diffusion with
//      noise level s and parameters (b,v,A) has exactly the same hitting
//      time distribution as a unit-noise process with (b/s, v/s, A/s) --
//      confirms the fix without re-deriving anything risky from scratch).
//   2. Removed the leftover `B ~ normal(...)` prior statement -- B is now
//      data (a fixed constant from SRDM), not a parameter, so a prior on
//      it did nothing (zero gradient contribution) but was confusing to
//      read.
//   3. v_correct/v_error/s_correct/s_error now sized N_CELLS (= 2*N_LEVELS)
//      instead of N_LEVELS, indexed directly by cell_correct/cell_false --
//      this is what makes them vary by (difficulty x instruction) jointly,
//      since cell already encodes both.

functions {
  real std_phi_pdf(real x) {
    return exp(-0.5 * x^2) / sqrt(2 * pi());
  }

  // Wald-with-uniform-start-point-variability PDF (Tillman et al. Eq. 5),
  // generalized to arbitrary noise level s (s=1 recovers the original).
  real rdm_pdf(real t, real b, real A, real v, real s) {
    real alpha = (b - A - t * v) / (s * sqrt(t));
    real beta  = (b - t * v) / (s * sqrt(t));
    real val = (1.0 / A) * (-v * Phi(alpha) + (s / sqrt(t)) * std_phi_pdf(alpha)
                             + v * Phi(beta)  - (s / sqrt(t)) * std_phi_pdf(beta));
    return fmax(val, 1e-12);
  }

  // CDF via fixed 20-point Gauss-Legendre quadrature of rdm_pdf, from ~0
  // to t (see the earlier file's header for why: the paper's own
  // closed-form Appendix A CDF could not be validated numerically).
  real rdm_survival_integral(real t, real b, real A, real v, real s) {
    array[20] real nodes = {-0.9931285992, -0.9639719273, -0.9122344283, -0.8391169718,
                             -0.7463319065, -0.6360536807, -0.5108670020, -0.3737060887,
                             -0.2277858511, -0.0765265211,  0.0765265211,  0.2277858511,
                              0.3737060887,  0.5108670020,  0.6360536807,  0.7463319065,
                              0.8391169718,  0.9122344283,  0.9639719273,  0.9931285992};
    array[20] real weights = {0.0176140071, 0.0406014298, 0.0626720483, 0.0832767416,
                               0.1019301198, 0.1181945320, 0.1316886384, 0.1420961093,
                               0.1491729865, 0.1527533871, 0.1527533871, 0.1491729865,
                               0.1420961093, 0.1316886384, 0.1181945320, 0.1019301198,
                               0.0832767416, 0.0626720483, 0.0406014298, 0.0176140071};
    real t_lo = 1e-6;
    real half_range = 0.5 * (t - t_lo);
    real mid = 0.5 * (t + t_lo);
    real cdf = 0;
    for (k in 1:20) {
      real x = half_range * nodes[k] + mid;
      cdf += half_range * weights[k] * rdm_pdf(x, b, A, v, s);
    }
    return fmin(fmax(cdf, 1e-12), 1 - 1e-12);
  }
}

data {
  int<lower=1> N_LEVELS;
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;
  array[N_correct] int<lower=1,upper=2*N_LEVELS> cell_correct;
  array[N_false]   int<lower=1,upper=2*N_LEVELS> cell_false;
  real<lower=0> t0_hi;
  real<lower=0> B;   // FIXED -- set this to SRDM's own fitted boundary value
}

transformed data {
  int N_CELLS = 2 * N_LEVELS;
}

parameters {
  real<lower=0> A;                        // starting-point range, shared
  vector<lower=0>[N_CELLS] v_correct;      // now varies by (difficulty x instruction)
  vector<lower=0>[N_CELLS] v_error;        // now varies by (difficulty x instruction)
  vector<lower=0>[N_CELLS] s_correct;      // now varies by (difficulty x instruction)
  vector<lower=0>[N_CELLS] s_error;        // now varies by (difficulty x instruction)
  real<lower=0, upper=t0_hi> t0;
}

model {
  A ~ normal(0.4, 0.3) T[0,];
  v_correct ~ normal(2.5, 1.5) T[0,];
  v_error ~ normal(1.0, 1.0) T[0,];
  s_correct ~ normal(1.0, 1.0) T[0,];
  s_error ~ normal(1.0, 1.0) T[0,];
  t0 ~ normal(0.25, 0.05);

  // Correct trials: correct accumulator wins (pdf), error accumulator loses (survival)
  {
    vector[N_correct] vc_trial = v_correct[cell_correct];
    vector[N_correct] ve_trial = v_error[cell_correct];
    vector[N_correct] sc_trial = s_correct[cell_correct];
    vector[N_correct] se_trial = s_error[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, vc_trial[i], sc_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, ve_trial[i], se_trial[i]);
      target += log(pdf_win) + log1m(cdf_lose);
    }
  }

  // Error trials: error accumulator wins, correct accumulator loses
  {
    vector[N_false] vc_trial = v_correct[cell_false];
    vector[N_false] ve_trial = v_error[cell_false];
    vector[N_false] sc_trial = s_correct[cell_false];
    vector[N_false] se_trial = s_error[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, ve_trial[i], se_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, vc_trial[i], sc_trial[i]);
      target += log(pdf_win) + log1m(cdf_lose);
    }
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] vc_trial = v_correct[cell_correct];
    vector[N_correct] ve_trial = v_error[cell_correct];
    vector[N_correct] sc_trial = s_correct[cell_correct];
    vector[N_correct] se_trial = s_error[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, vc_trial[i], sc_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, ve_trial[i], se_trial[i]);
      log_lik[i] = log(pdf_win) + log1m(cdf_lose);
    }
  }
  {
    vector[N_false] vc_trial = v_correct[cell_false];
    vector[N_false] ve_trial = v_error[cell_false];
    vector[N_false] sc_trial = s_correct[cell_false];
    vector[N_false] se_trial = s_error[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, ve_trial[i], se_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, vc_trial[i], sc_trial[i]);
      log_lik[N_correct + i] = log(pdf_win) + log1m(cdf_lose);
    }
  }
}
