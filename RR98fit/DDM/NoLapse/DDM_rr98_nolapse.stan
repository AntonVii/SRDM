// DDM for rr98, NO-LAPSE variant.
// N_LEVELS is passed in as data — set it to 7, 11, 16, 33, whatever you want.
//
// Difference from DDM_rr98_configurable.stan:
//   - No p_lapse parameter, no lapse mixture, no t0-blend.
//   - Relies instead on the data having been trimmed (fastest/slowest 1%
//     per participant x instruction x difficulty cell removed) and on
//     t0_hi being set to min(RT) over that trimmed data. Since t0 has a
//     hard upper bound of t0_hi, t_dec = rt - t0 > 0 for every trial by
//     construction, so no blend is needed at t0.

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
  int J = 15;
  int N_CELLS = 2 * N_LEVELS;

  array[N_CELLS] int sat_of;
  array[N_CELLS] int diff_of;
  for (s in 1:2)
    for (l in 1:N_LEVELS) {
      int idx = (s-1)*N_LEVELS + l;
      sat_of[idx] = s;
      diff_of[idx] = l;
    }

  vector[5] gl_nodes  = [-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985]';
  vector[5] gl_weight = [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]';
  vector[5] gl_logw;
  for (k in 1:5) gl_logw[k] = log(gl_weight[k]) - log(2.0);
}

parameters {
  vector<lower=0>[2] a;
  vector<lower=0>[N_LEVELS] v_base;
  real<lower=0> sv;
  real<lower=0, upper=0.9> sz;
  real<lower=0, upper=t0_hi> t0;
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

  // Correct trials
  {
    vector[N_correct] a_trial = a_full[cell_correct];
    vector[N_correct] v_trial = v_full[cell_correct];

    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, v_trial[i], sv, a_trial[i], 1 - w_node, J);
      }
      target += log_sum_exp(terms);
    }
  }

  // Error trials
  {
    vector[N_false] a_trial = a_full[cell_false];
    vector[N_false] v_trial = v_full[cell_false];

    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, -v_trial[i], sv, a_trial[i], w_node, J);
      }
      target += log_sum_exp(terms);
    }
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] a_trial = a_full[cell_correct];
    vector[N_correct] v_trial = v_full[cell_correct];
    for (i in 1:N_correct) {
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, v_trial[i], sv, a_trial[i], 1 - w_node, J);
      }
      log_lik[i] = log_sum_exp(terms);
    }
  }
  {
    vector[N_false] a_trial = a_full[cell_false];
    vector[N_false] v_trial = v_full[cell_false];
    for (i in 1:N_false) {
      real t_dec = fmax(rt_false[i] - t0, 1e-4);
      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, -v_trial[i], sv, a_trial[i], w_node, J);
      }
      log_lik[N_correct + i] = log_sum_exp(terms);
    }
  }
}
