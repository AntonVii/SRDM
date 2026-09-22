// DDM matching the rtdists (Singmann et al.) reanalysis scheme for RR98.
//
// Key differences from our other DDM variants:
//   - Models raw response identity ("dark" vs "light") directly, not
//     correctness-recoded trials. Each stimulus bin gets ONE signed
//     drift rate v[i] (can be positive or negative), rather than a
//     positive-only drift plus a correctness-based sign flip.
//   - Includes z (relative starting point / response bias) as a free
//     parameter, matching rtdists. Our other variants hardcoded z=0.5*a
//     (no bias) since we were working with correctness, where bias
//     doesn't have a well-defined sign; here it does.
//   - Meant to be fit SEPARATELY per participant x instruction condition
//     (i.e. one Stan run per subset), matching rtdists exactly -- no
//     shared boundary/drift structure across conditions within one fit.
//   - No lapse (uses the same trim + hard-t0-bound approach as our other
//     no-lapse variants).
//
// "Dark" = upper boundary, "light" = lower boundary (arbitrary but fixed
// convention -- matches the direction rtdists uses).

functions {
  real bkg_log_density(real t, real nu, real eta, real a, real w, int J) {
    real log_prefactor = -0.5 * log(t^3 * (1 + eta^2 * t))
      - (nu^2 * t - 2*nu*a*w + eta^2 * square(a*w)) / (2 * (1 + eta^2 * t));

    real total = 0;
    real sign = 1;
    for (j in 0:J) {
      real r_j;
      if (j % 2 == 0) r_j = j*a + a*w;
      else             r_j = j*a + a*(1 - w);
      real phi_val = exp(-0.5 * square(r_j / sqrt(t))) / sqrt(2 * pi());
      total += sign * r_j * phi_val;
      sign = -sign;
    }
    return log_prefactor + log(fmax(total, 1e-300));
  }
}

data {
  int<lower=1> N_LEVELS;      // number of stimulus bins (5, to match rtdists)
  int N_dark;
  int N_light;
  vector[N_dark]  rt_dark;
  vector[N_light] rt_light;
  array[N_dark]  int<lower=1,upper=N_LEVELS> level_dark;
  array[N_light] int<lower=1,upper=N_LEVELS> level_light;
  real<lower=0> t0_hi;        // min(RT) over TRIMMED data, this participant x condition
}

transformed data {
  int J = 15;
  vector[5] gl_nodes  = [-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985]';
  vector[5] gl_weight = [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]';
  vector[5] gl_logw;
  for (k in 1:5) gl_logw[k] = log(gl_weight[k]) - log(2.0);
}

parameters {
  real<lower=0> a;
  vector[N_LEVELS] v;                  // signed drift, one per stimulus bin
  real<lower=0.1, upper=0.9> z_rel;    // relative starting point (bias)
  real<lower=0> sv;
  real<lower=0, upper=0.9> sz;
  real<lower=0, upper=t0_hi> t0;
}

model {
  a ~ normal(1.5, 0.75);
  v ~ normal(0, 2.5);
  z_rel ~ normal(0.5, 0.15);
  sv ~ normal(0.6, 0.3) T[0,];
  sz ~ beta(2, 8);
  t0 ~ normal(0.25, 0.05);

  // "Dark" responses: hit upper boundary directly -> (v[i], z_rel)
  {
    vector[N_dark] v_trial = v[level_dark];
    for (i in 1:N_dark) {
      real t_dec = fmax(rt_dark[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = z_rel + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz] + bkg_log_density(t_dec, v_trial[i], sv, a, w_node, J);
      }
      target += log_sum_exp(terms);
    }
  }

  // "Light" responses: hit lower boundary -> mirror via (-v[i], 1 - z_rel)
  {
    vector[N_light] v_trial = v[level_light];
    for (i in 1:N_light) {
      real t_dec = fmax(rt_light[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = z_rel + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz] + bkg_log_density(t_dec, -v_trial[i], sv, a, 1 - w_node, J);
      }
      target += log_sum_exp(terms);
    }
  }
}

generated quantities {
  vector[N_dark + N_light] log_lik;
  {
    vector[N_dark] v_trial = v[level_dark];
    for (i in 1:N_dark) {
      real t_dec = fmax(rt_dark[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = z_rel + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz] + bkg_log_density(t_dec, v_trial[i], sv, a, w_node, J);
      }
      log_lik[i] = log_sum_exp(terms);
    }
  }
  {
    vector[N_light] v_trial = v[level_light];
    for (i in 1:N_light) {
      real t_dec = fmax(rt_light[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = z_rel + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz] + bkg_log_density(t_dec, -v_trial[i], sv, a, 1 - w_node, J);
      }
      log_lik[N_dark + i] = log_sum_exp(terms);
    }
  }
}
