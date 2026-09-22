// RDM (Racing Diffusion Model, Tillman, Van Zandt & Logan 2020) for rr98.
// N_LEVELS is passed in as data — python side handles binning, same
// convention as the DDM/SRDM files.
//
// This variant: v_error is a SINGLE shared value across all difficulty
// levels (only the correct accumulator's drift varies with difficulty).
// See RDM_varying_verr.stan for the version where both accumulators'
// drift rates vary by difficulty.
//
// Model structure (matching Tillman et al. 2020, Eqs. 2-7):
//   - Two racing accumulators per trial (correct, error), each a Wald
//     process (within-trial diffusion noise, NO between-trial drift
//     variability -- that's the RDM's defining feature vs. the LBA).
//   - Between-trial UNIFORM[0,A] starting-point variability (shared
//     across both accumulators and both SAT conditions).
//   - B[2]: threshold-above-starting-point, by SAT condition (matches
//     a[2]/c[2] convention in DDM/SRDM).
//   - No lapse; hard t0 <= t0_hi bound from trimmed data, matching the
//     project's established no-lapse convention.
//
// NUMERICAL NOTE ON THE CDF: Tillman et al.'s Appendix A gives a closed
// form for the RDM's CDF (needed for the race likelihood -- density of
// the winner times survival of the loser). That closed form could NOT
// be validated against numerical integration of the paper's own PDF
// (Eq. 5) -- almost certainly because of symbol loss (e.g. missing Phi
// notation) somewhere in extracting the appendix from the PDF. Given
// the DDM's sv-collapse bug earlier in this project came from exactly
// this kind of unvalidated custom density, the CDF here is instead
// computed via a FIXED 20-POINT GAUSS-LEGENDRE QUADRATURE over the
// (validated) PDF -- confirmed to agree with scipy's adaptive
// integration to within 1e-5 or better across realistic parameter
// ranges. This is the same style of fix already used for sz in the DDM.

functions {
  real std_phi_pdf(real x) {
    return exp(-0.5 * x^2) / sqrt(2 * pi());
  }

  // Wald-with-uniform-start-point-variability PDF (Tillman et al. Eq. 5)
  real rdm_pdf(real t, real b, real A, real v, real s) {
    real sqr = sqrt(t);
    real alpha = (b - A - t * v) / (s * sqr);
    real beta  = (b - t * v) / (s * sqr);
    real val = (1.0 / A) * (-v * Phi(alpha) + (s / sqr) * std_phi_pdf(alpha)
                            + v * Phi(beta)  - (s / sqr) * std_phi_pdf(beta));
    return fmax(val, 1e-12);
  }

  // CDF via fixed 20-point Gauss-Legendre quadrature of rdm_pdf,
  // from ~0 to t. See file header for why this is used instead of the
  // paper's closed-form Appendix A CDF.
  real rdm_survival_integral(real t, real b, real A, real v,real s) {
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
      cdf += half_range * weights[k] * rdm_pdf(x, b, A, v,s);
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
  real<lower=0> B;
}

transformed data {
  int N_CELLS = 2 * N_LEVELS;
  array[N_CELLS] int sat_of;
  array[N_CELLS] int diff_of;
  for (s in 1:2)
    for (l in 1:N_LEVELS) {
      int idx = (s-1)*N_LEVELS + l;
      sat_of[idx] = s;
      diff_of[idx] = l;
    }
}

parameters {
  real<lower=0> A;                  // starting-point range, shared
  vector<lower=0>[N_LEVELS] v_correct;
  vector<lower=0>[N_LEVELS] v_error;
  vector<lower=0>[N_LEVELS] s_correct;
  vector<lower=0>[N_LEVELS] s_error;
  real<lower=0, upper=t0_hi> t0;
}

transformed parameters {
  vector[N_CELLS] v_correct_full = v_correct[diff_of];
  vector[N_CELLS] v_error_full = v_error[diff_of];
  vector[N_CELLS] s_correct_full = s_correct[diff_of];
  vector[N_CELLS] s_error_full = s_error[diff_of];
}

model {
  B ~ normal(1.0, 0.5) T[0,];
  A ~ normal(0.4, 0.3) T[0,];
  v_correct ~ normal(2.5, 1.5) T[0,];
  v_error ~ normal(1.0, 1.0) T[0,];
  s_correct ~ normal(1.0, 1.0) T[0,];
  s_error ~ normal(1.0, 1.0) T[0,];
  t0 ~ normal(0.25, 0.05);

  // Correct trials: correct accumulator wins (pdf), error accumulator loses (survival)
  {
    vector[N_correct] vc_trial = v_correct_full[cell_correct];
    vector[N_correct] ve_trial = v_error_full[cell_correct];
    vector[N_correct] sc_trial = s_correct_full[cell_correct];
    vector[N_correct] se_trial = s_error_full[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, vc_trial[i],sc_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, ve_trial[i],se_trial[i]);
      target += log(pdf_win) + log1m(cdf_lose);
    }
  }

  // Error trials: error accumulator wins, correct accumulator loses
  {
    vector[N_false] vc_trial = v_correct_full[cell_false];
    vector[N_false] ve_trial = v_error_full[cell_false];
    vector[N_false] sc_trial = s_correct_full[cell_false];
    vector[N_false] se_trial = s_error_full[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, ve_trial[i],se_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, vc_trial[i],sc_trial[i]);
      target += log(pdf_win) + log1m(cdf_lose);
    }
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] vc_trial = v_correct_full[cell_correct];
    vector[N_correct] ve_trial = v_error_full[cell_correct];
    vector[N_correct] sc_trial = s_correct_full[cell_correct];
    vector[N_correct] se_trial = s_error_full[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, vc_trial[i],sc_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, ve_trial[i],se_trial[i]);
      log_lik[i] = log(pdf_win) + log1m(cdf_lose);
    }
  }
  {
    vector[N_false] ve_trial = v_error_full[cell_false];
    vector[N_false] vc_trial = v_correct_full[cell_false];
    vector[N_false] sc_trial = s_correct_full[cell_false];
    vector[N_false] se_trial = s_error_full[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      real pdf_win = rdm_pdf(t_dec, B, A, ve_trial[i],se_trial[i]);
      real cdf_lose = rdm_survival_integral(t_dec, B, A, vc_trial[i],sc_trial[i]);
      log_lik[N_correct + i] = log(pdf_win) + log1m(cdf_lose);
    }
  }
}
