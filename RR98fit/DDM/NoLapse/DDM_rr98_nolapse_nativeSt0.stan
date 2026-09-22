// DDM for rr98, NO-LAPSE variant -- NATIVE WIENER_LPDF VERSION.
// N_LEVELS is passed in as data — set it to 7, 11, 16, 33, whatever you want.
//
// Difference from the previous DDM_rr98_nolapse.stan:
//   - Replaced the custom bkg_log_density() function (Blurton et al. 2017
//     closed-form, with sv integrated out analytically and sz handled by
//     a bolted-on 5-point Gauss-Legendre quadrature) with Stan Math's own
//     native wiener_lpdf(), which handles sv AND sz internally using the
//     same numerical lineage (Navarro & Fuss 2009 / Blurton et al. 2017 /
//     Foster & Singmann 2021) that fast-dm/rtdists use, written in part
//     by Andreas Voss himself. Confirmed empirically that this recovers
//     non-degenerate sv (0.08-1.05 range) where the old custom density
//     collapsed sv to ~0.001-0.01 on the same data -- the old closed-form
//     sv-integration was suppressing it, not the data.
//   - No manual GL quadrature over sz needed any more -- sz is passed
//     directly as wiener_lpdf's `sw` argument and integrated internally.
//   - No bias parameter: correct/error trials both use w = 0.5 (this
//     model was never estimating starting-point bias, only its
//     variability), so 1-w = w = 0.5 for both boundaries -- consistent
//     with the previous code's implicit convention.
//   - Parameterization otherwise UNCHANGED from before: shared (not
//     condition-specific) sv and sz, a[2] by SAT condition, v_base by
//     difficulty level, no lapse, hard t0 <= t0_hi bound from trimmed data.
//
// REQUIRES CmdStan >= ~2.36 (wiener_lpdf's 7-argument overload isn't in
// 2.35.0 or earlier). Confirmed working against CmdStan 2.39.0.

data {
  int<lower=1> N_LEVELS;   // number of difficulty bins (7, 11, 16, 33, ...)
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;
  array[N_correct] int<lower=1,upper=2*N_LEVELS> cell_correct;
  array[N_false]   int<lower=1,upper=2*N_LEVELS> cell_false;
  real<lower=0> t0_hi;     // min(RT) over TRIMMED data for this participant
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
  vector<lower=0>[2] a;
  vector<lower=0>[N_LEVELS] v_base;
  real<lower=0> sv;
  real<lower=0, upper=0.9> sz;   // stays comfortably below the sw<1 constraint at w=0.5
  real<lower=0, upper=t0_hi> t0;
  real<lower=0, upper=0.2> st0;
}

transformed parameters {
  vector[N_CELLS] a_full = a[sat_of];
  vector[N_CELLS] v_full = v_base[diff_of];
}

model {
  a ~ normal(1.5, 0.75);
  v_base ~ normal(2.0, 1.5);
  sv ~ normal(0.2, 0.525) T[0,];
  sz ~ beta(10, 10);
  t0 ~ normal(0.25, 0.05);
  st0 ~ normal(0.05, 0.03) T[0,];

  // Correct trials: upper boundary, w = 0.5
  {
    vector[N_correct] a_trial = a_full[cell_correct];
    vector[N_correct] v_trial = v_full[cell_correct];
    for (i in 1:N_correct)
      target += wiener_lpdf(rt_correct[i] | a_trial[i], t0, 0.5, v_trial[i], sv, sz, st0);
  }

  // Error trials: mirror -> lower boundary via (-v, 1-w); since w=0.5, 1-w=0.5 too
  {
    vector[N_false] a_trial = a_full[cell_false];
    vector[N_false] v_trial = v_full[cell_false];
    for (i in 1:N_false)
      target += wiener_lpdf(rt_false[i] | a_trial[i], t0, 0.5, -v_trial[i], sv, sz, st0);
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] a_trial = a_full[cell_correct];
    vector[N_correct] v_trial = v_full[cell_correct];
    for (i in 1:N_correct)
      log_lik[i] = wiener_lpdf(rt_correct[i] | a_trial[i], t0, 0.5, v_trial[i], sv, sz, st0);
  }
  {
    vector[N_false] a_trial = a_full[cell_false];
    vector[N_false] v_trial = v_full[cell_false];
    for (i in 1:N_false)
      log_lik[N_correct + i] = wiener_lpdf(rt_false[i] | a_trial[i], t0, 0.5, -v_trial[i], sv, sz, st0);
  }
}
