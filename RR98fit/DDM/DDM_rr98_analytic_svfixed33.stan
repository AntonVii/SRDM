functions {
  // Blurton, Kesselmeier & Gondan (2017, J. Math. Psych.), Eq. 1:
  // analytic first-passage-time density at the LOWER barrier of a
  // two-boundary Wiener process with drift rate ~ Normal(nu, eta^2)
  // across trials. Barriers at 0 and a; starting point at a*w, 0<w<1.
  // Series truncated at a FIXED J terms (not adaptive - keeps autodiff
  // well-behaved). J=20 is comfortably above the ~8 terms the paper's
  // own Table 1 found sufficient for a, t ranges matching this model.
  real bkg_log_density(real t, real nu, real eta, real a, real w, int J) {
    real log_prefactor = -0.5 * log(t^3 * (1 + eta^2 * t))
      - (nu^2 * t - 2*nu*a*w + eta^2 * square(a*w)) / (2 * (1 + eta^2 * t));

    real total = 0;
    real sign = 1;
    for (j in 0:J) {
      real r_j;
      if (j % 2 == 0) {
        r_j = j*a + a*w;
      } else {
        r_j = j*a + a*(1 - w);
      }
      real phi_val = exp(-0.5 * square(r_j / sqrt(t))) / sqrt(2 * pi());
      total += sign * r_j * phi_val;
      sign = -sign;
    }
    // guard against the alternating sum dipping to ~0 or slightly negative
    // due to truncation/floating-point error (should not happen with J=20
    // given the paper's convergence guarantees, but cheap to protect against)
    return log_prefactor + log(fmax(total, 1e-300));
  }
}

data {
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;

  // combined SAT x difficulty cell index, values in 1..14
  array[N_correct] int<lower=1,upper=66> cell_correct;
  array[N_false] int<lower=1,upper=66> cell_false;

  real<lower=0> max_rt;
  real<lower=0> t0_hi;
}

transformed data {
  real t0_blend_width = 0.03;
  int J = 20;  // fixed series truncation - see note above

  array[66] int sat_of;
  array[66] int diff_of;
  for (s in 1:2) {
    for (l in 1:33) {
      int idx = (s-1)*33 + l;
      sat_of[idx] = s;
      diff_of[idx] = l;
    }
  }

  // 5-point Gauss-Legendre quadrature on [-1,1], for integrating
  // w ~ Uniform(0.5 - sz/2, 0.5 + sz/2). Standard tabulated values.
  vector[5] gl_nodes  = [-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985]';
  vector[5] gl_weight = [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]';
  vector[5] gl_logw;
  for (k in 1:5) gl_logw[k] = log(gl_weight[k]) - log(2.0);
}

parameters {
  vector<lower=0>[2] a;              // boundary separation x instruction
  vector<lower=0>[33] v_base;         // drift rate x stimulus difficulty
  real<lower=0> sv;                  // across-trial drift variability (analytic)
  real<lower=0, upper=0.9> sz;       // across-trial starting-point variability (quadrature)
  real<lower=0, upper=t0_hi> t0;     // non-decision time, single scalar
  real<lower=0,upper=1> p_lapse;     // lapse rate, single scalar
}

transformed parameters {
  vector[66] a_full = a[sat_of];
  vector[66] v_full = v_base[diff_of];
}

model {
  a ~ normal(0.15, 0.1);
  v_base ~ normal(0, 1);
  sv ~ normal(0.06,0.02);
  sz ~ beta(5, 8);
  t0 ~ normal(0.15, 0.1);
  p_lapse ~ beta(1, 50);

  // -----------------------
  // Correct trials (upper boundary): use nu -> -nu, w -> 1-w substitution
  // -----------------------
  {
    vector[N_correct] a_trial = a_full[cell_correct];
    vector[N_correct] v_trial = v_full[cell_correct];
    vector[N_correct] lapse_lp = rep_vector(-log(max_rt), N_correct);

    for (i in 1:N_correct) {
      real t_raw = rt_correct[i] - t0;
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);

      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, v_trial[i], sv, a_trial[i], 1 - w_node, J);
      }
      real wiener_lp = log_sum_exp(terms);

      real log_w   = log_inv_logit(t_raw / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw / t0_blend_width);
      target += log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], wiener_lp),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }

  // -----------------------
  // False trials (lower boundary): direct formula
  // -----------------------
  {
    vector[N_false] a_trial = a_full[cell_false];
    vector[N_false] v_trial = v_full[cell_false];
    vector[N_false] lapse_lp = rep_vector(-log(max_rt), N_false);

    for (i in 1:N_false) {
      real t_raw = rt_false[i] - t0;
      real t_dec = fmax(rt_false[i] - t0, 1e-4);

      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, -v_trial[i], sv, a_trial[i], w_node, J);
      }
      real wiener_lp = log_sum_exp(terms);

      real log_w   = log_inv_logit(t_raw / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw / t0_blend_width);
      target += log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], wiener_lp),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;

  {
    vector[N_correct] a_trial = a_full[cell_correct];
    vector[N_correct] v_trial = v_full[cell_correct];
    vector[N_correct] lapse_lp = rep_vector(-log(max_rt), N_correct);

    for (i in 1:N_correct) {
      real t_raw = rt_correct[i] - t0;
      real t_dec = fmax(rt_correct[i] - t0, 1e-4);

      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, v_trial[i], sv, a_trial[i], 1 - w_node, J);
      }
      real wiener_lp = log_sum_exp(terms);

      real log_w   = log_inv_logit(t_raw / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw / t0_blend_width);
      log_lik[i] = log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], wiener_lp),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }

  {
    vector[N_false] a_trial = a_full[cell_false];
    vector[N_false] v_trial = v_full[cell_false];
    vector[N_false] lapse_lp = rep_vector(-log(max_rt), N_false);

    for (i in 1:N_false) {
      real t_raw = rt_false[i] - t0;
      real t_dec = fmax(rt_false[i] - t0, 1e-4);

      vector[5] terms;
      for (kz in 1:5) {
        real w_node = 0.5 + (sz/2) * gl_nodes[kz];
        terms[kz] = gl_logw[kz]
          + bkg_log_density(t_dec, -v_trial[i], sv, a_trial[i], w_node, J);
      }
      real wiener_lp = log_sum_exp(terms);

      real log_w   = log_inv_logit(t_raw / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw / t0_blend_width);
      log_lik[N_correct + i] = log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], wiener_lp),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }
}
