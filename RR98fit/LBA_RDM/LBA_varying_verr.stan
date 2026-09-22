// LBA (Linear Ballistic Accumulator, Brown & Heathcote 2008) for rr98.
// N_LEVELS is passed in as data, same convention as the other model files.
//
// This variant: v_error VARIES BY DIFFICULTY LEVEL, same as v_correct.
// See LBA_shared_verr.stan for the version where v_error is a single
// shared value across all difficulty levels.
//
// pdf/cdf functions are a direct port of Annis, Miller & Palmeri's (2016)
// reference Stan implementation (github.com/jeff324/Stan_LBA), which is
// itself the standard companion code for their Stan tutorial paper --
// verified here against numerical integration to machine precision
// before use (unlike the RDM's CDF, which needed a different approach --
// see RDM_shared_verr.stan header).
//
// IDENTIFIABILITY: v, s (between-trial drift SD), and b are only
// jointly identified up to a common scale factor. Per your instruction,
// this is resolved exactly as the reference implementation does: s is
// FIXED at 1 (transformed data, not a parameter). This still gives
// genuine between-trial drift variability (the model is not
// deterministic) -- s=1 is a fixed known value, not zero.
//
// No lapse; hard t0 <= t0_hi bound from trimmed data, matching the
// project's established no-lapse convention. Threshold is parameterized
// as b = A + k (k by SAT condition), keeping the starting-point range
// mechanically below threshold, matching both the reference code and
// Tillman et al.'s B = b - A convention.

functions {
  real lba_pdf(real t, real b, real A, real v, real s) {
    real b_A_tv_ts = (b - A - t*v) / (t*s);
    real b_tv_ts   = (b - t*v) / (t*s);
    real term1 = v * Phi(b_A_tv_ts);
    real term2 = s * exp(std_normal_lpdf(b_A_tv_ts));
    real term3 = v * Phi(b_tv_ts);
    real term4 = s * exp(std_normal_lpdf(b_tv_ts));
    return fmax((1.0/A) * (-term1 + term2 + term3 - term4), 1e-12);
  }

  real lba_survival_integral(real t, real b, real A, real v, real s) {
    real b_A_tv = b - A - t*v;
    real b_tv = b - t*v;
    real ts = t*s;
    real term1 = b_A_tv/A * Phi(b_A_tv/ts);
    real term2 = b_tv/A * Phi(b_tv/ts);
    real term3 = ts/A * exp(std_normal_lpdf(b_A_tv/ts));
    real term4 = ts/A * exp(std_normal_lpdf(b_tv/ts));
    real cdf = 1 + term1 - term2 + term3 - term4;
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
}

transformed data {
  int N_CELLS = 2 * N_LEVELS;
  real s = 1.0;   // FIXED, per reference implementation's identifiability solution
  array[N_CELLS] int sat_of;
  array[N_CELLS] int diff_of;
  for (sc in 1:2)
    for (l in 1:N_LEVELS) {
      int idx = (sc-1)*N_LEVELS + l;
      sat_of[idx] = sc;
      diff_of[idx] = l;
    }
}

parameters {
  vector<lower=0>[2] k;              // threshold increment above A, by SAT
  real<lower=0> A;                   // starting-point range, shared
  vector<lower=0>[N_LEVELS] v_correct;
  vector<lower=0>[N_LEVELS] v_error; // VARIES by difficulty level, like v_correct
  real<lower=0, upper=t0_hi> t0;
}

transformed parameters {
  vector[N_CELLS] b_full = A + k[sat_of];
  vector[N_CELLS] v_correct_full = v_correct[diff_of];
  vector[N_CELLS] v_error_full = v_error[diff_of];
}

model {
  k ~ normal(0.6, 0.5) T[0,];
  A ~ normal(0.4, 0.3) T[0,];
  v_correct ~ normal(2.5, 1.5) T[0,];
  v_error ~ normal(1.0, 1.0) T[0,];  // same prior, now per-level
  t0 ~ normal(0.25, 0.05);

  {
    vector[N_correct] b_trial = b_full[cell_correct];
    vector[N_correct] vc_trial = v_correct_full[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      real pdf_win = lba_pdf(t_dec, b_trial[i], A, vc_trial[i], s);
      real cdf_lose = lba_survival_integral(t_dec, b_trial[i], A, v_error_full[cell_correct[i]], s);
      real prob_neg = Phi(-vc_trial[i]/s) * Phi(-v_error_full[cell_correct[i]]/s);
      target += log(pdf_win) + log1m(cdf_lose) - log1m(prob_neg);
    }
  }
  {
    vector[N_false] b_trial = b_full[cell_false];
    vector[N_false] vc_trial = v_correct_full[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      real pdf_win = lba_pdf(t_dec, b_trial[i], A, v_error_full[cell_false[i]], s);
      real cdf_lose = lba_survival_integral(t_dec, b_trial[i], A, vc_trial[i], s);
      real prob_neg = Phi(-vc_trial[i]/s) * Phi(-v_error_full[cell_false[i]]/s);
      target += log(pdf_win) + log1m(cdf_lose) - log1m(prob_neg);
    }
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] b_trial = b_full[cell_correct];
    vector[N_correct] vc_trial = v_correct_full[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      real pdf_win = lba_pdf(t_dec, b_trial[i], A, vc_trial[i], s);
      real cdf_lose = lba_survival_integral(t_dec, b_trial[i], A, v_error_full[cell_correct[i]], s);
      real prob_neg = Phi(-vc_trial[i]/s) * Phi(-v_error_full[cell_correct[i]]/s);
      log_lik[i] = log(pdf_win) + log1m(cdf_lose) - log1m(prob_neg);
    }
  }
  {
    vector[N_false] b_trial = b_full[cell_false];
    vector[N_false] vc_trial = v_correct_full[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      real pdf_win = lba_pdf(t_dec, b_trial[i], A, v_error_full[cell_false[i]], s);
      real cdf_lose = lba_survival_integral(t_dec, b_trial[i], A, vc_trial[i], s);
      real prob_neg = Phi(-vc_trial[i]/s) * Phi(-v_error_full[cell_false[i]]/s);
      log_lik[N_correct + i] = log(pdf_win) + log1m(cdf_lose) - log1m(prob_neg);
    }
  }
}
