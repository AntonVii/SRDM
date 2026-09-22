// Same reanalysis-style scheme as DDM_rr98_reanalysis_style.stan (5 bins,
// fit separately per participant x instruction, raw response coding,
// signed per-bin drift, free starting-point bias) -- but using Stan's
// OWN NATIVE wiener_lpdf() function instead of our hand-written Blurton
// closed-form density.
//
// wiener_lpdf(y | a, t0, w, v, sv, sw, st0) is Stan Math's built-in
// 7-parameter Wiener first-passage-time density (Navarro & Fuss 2009 /
// Blurton et al. 2017 / Foster & Singmann 2021 methods, adaptively
// switched internally), added to Stan Math specifically to bring the
// same numerical lineage used by fast-dm/rtdists (Voss & Voss) natively
// into Stan. If THIS gives a materially different sv than our custom
// code, the difference is in the density implementation itself, not in
// binning, response coding, trimming, or optimizer restarts -- all of
// which we've already ruled out.
//
// Requires CmdStan >= 2.36ish (built and confirmed working against
// CmdStan 2.39.0 in this project; not available in 2.35.0).

data {
  int<lower=1> N_LEVELS;
  int N_dark;
  int N_light;
  vector[N_dark]  rt_dark;
  vector[N_light] rt_light;
  array[N_dark]  int<lower=1,upper=N_LEVELS> level_dark;
  array[N_light] int<lower=1,upper=N_LEVELS> level_light;
  real<lower=0> t0_hi;
}

parameters {
  real<lower=0> a;
  vector[N_LEVELS] v;
  real<lower=0.1, upper=0.9> z_rel;
  real<lower=0> sv;
  // sw (starting-point variability) must satisfy sw < min(2*w, 2*(1-w))
  // for the native function's internal constraints -- keep it comfortably
  // inside that range via the upper bound below.
  real<lower=0, upper=0.9> sz;
  real<lower=0, upper=t0_hi> t0;
}

transformed parameters {
  real sw = fmin(sz, 0.98 * fmin(2 * z_rel, 2 * (1 - z_rel)));
}

model {
  a ~ normal(1.5, 0.75);
  v ~ normal(0, 2.5);
  z_rel ~ normal(0.5, 0.15);
  sv ~ normal(0.6, 0.3) T[0,];
  sz ~ beta(2, 8);
  t0 ~ normal(0.25, 0.05);

  for (i in 1:N_dark)
    target += wiener_lpdf(rt_dark[i] | a, t0, z_rel, v[level_dark[i]], sv, sw, 0);
  for (i in 1:N_light)
    target += wiener_lpdf(rt_light[i] | a, t0, 1 - z_rel, -v[level_light[i]], sv, sw, 0);
}

generated quantities {
  vector[N_dark + N_light] log_lik;
  for (i in 1:N_dark)
    log_lik[i] = wiener_lpdf(rt_dark[i] | a, t0, z_rel, v[level_dark[i]], sv, sw, 0);
  for (i in 1:N_light)
    log_lik[N_dark + i] = wiener_lpdf(rt_light[i] | a, t0, 1 - z_rel, -v[level_light[i]], sv, sw, 0);
}
