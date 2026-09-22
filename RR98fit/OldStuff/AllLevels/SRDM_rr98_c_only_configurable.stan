// SRDM (c_only) for rr98 with CONFIGURABLE number of difficulty levels.
// N_LEVELS is passed in as data — set it to 7, 11, 16, 33, whatever you want.

data {
  int<lower=1> N_LEVELS;
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;
  array[N_correct] int<lower=1,upper=2*N_LEVELS> cell_correct;
  array[N_false]   int<lower=1,upper=2*N_LEVELS> cell_false;
  real<lower=0> max_rt;
  real<lower=0> t0_hi;
}

transformed data {
  real t0_blend_width = 0.03;
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
  vector[2] c;
  real<lower=0> B;
  real<lower=0, upper=t0_hi> t0;
  vector<lower=0>[N_LEVELS] d_base;
  real<lower=0.05, upper=50> r;
  real<lower=0, upper=1> p_lapse;
}

transformed parameters {
  vector[N_CELLS] c_full = c[sat_of];
  vector[N_CELLS] B_full = rep_vector(B, N_CELLS);
  vector[N_CELLS] d_full = d_base[diff_of];

  vector[N_CELLS] pk = fmin(fmax(Phi(d_full/2 - c_full), 1e-6), 1 - 1e-6);
  vector[N_CELLS] pf = fmin(fmax(Phi(-d_full/2 - c_full), 1e-6), 1 - 1e-6);
  vector[N_CELLS] vk = pk .* r;
  vector[N_CELLS] vf = pf .* r;
  vector[N_CELLS] sigma_k = fmax(sqrt(pk .* (1 - pk) * r), 0.05);
  vector[N_CELLS] sigma_f = fmax(sqrt(pf .* (1 - pf) * r), 0.05);
}

model {
  t0 ~ normal(0.2, 0.15);
  d_base ~ normal(1.5, 0.5);
  c ~ normal(0, 0.5);
  B ~ exponential(1);
  r ~ gamma(9, 2);
  p_lapse ~ beta(1, 50);

  // Correct trials
  {
    vector[N_correct] B_trial = B_full[cell_correct];
    vector[N_correct] own_v   = vk[cell_correct];
    vector[N_correct] oth_v   = vf[cell_correct];
    vector[N_correct] own_sig = sigma_k[cell_correct];
    vector[N_correct] oth_sig = sigma_f[cell_correct];
    vector[N_correct] lapse_lp = rep_vector(-log(max_rt), N_correct);

    vector[N_correct] t_raw  = rt_correct - t0;
    vector[N_correct] t_safe = fmax(t_raw, 1e-4);

    vector[N_correct] f_win =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));
    vector[N_correct] F_lose =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));

    vector[N_correct] sdt_lp = log(f_win) + log1m(F_lose);
    for (i in 1:N_correct) {
      real log_w   = log_inv_logit(t_raw[i] / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw[i] / t0_blend_width);
      target += log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], sdt_lp[i]),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }

  // Error trials
  {
    vector[N_false] B_trial = B_full[cell_false];
    vector[N_false] own_v   = vf[cell_false];
    vector[N_false] oth_v   = vk[cell_false];
    vector[N_false] own_sig = sigma_f[cell_false];
    vector[N_false] oth_sig = sigma_k[cell_false];
    vector[N_false] lapse_lp = rep_vector(-log(max_rt), N_false);

    vector[N_false] t_raw  = rt_false - t0;
    vector[N_false] t_safe = fmax(t_raw, 1e-4);

    vector[N_false] f_win =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));
    vector[N_false] F_lose =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));

    vector[N_false] sdt_lp = log(f_win) + log1m(F_lose);
    for (i in 1:N_false) {
      real log_w   = log_inv_logit(t_raw[i] / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw[i] / t0_blend_width);
      target += log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], sdt_lp[i]),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] B_trial = B_full[cell_correct];
    vector[N_correct] own_v   = vk[cell_correct];
    vector[N_correct] oth_v   = vf[cell_correct];
    vector[N_correct] own_sig = sigma_k[cell_correct];
    vector[N_correct] oth_sig = sigma_f[cell_correct];
    vector[N_correct] lapse_lp = rep_vector(-log(max_rt), N_correct);
    vector[N_correct] t_raw  = rt_correct - t0;
    vector[N_correct] t_safe = fmax(t_raw, 1e-4);
    vector[N_correct] f_win =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));
    vector[N_correct] F_lose =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));
    vector[N_correct] sdt_lp = log(f_win) + log1m(F_lose);
    for (i in 1:N_correct) {
      real log_w   = log_inv_logit(t_raw[i] / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw[i] / t0_blend_width);
      log_lik[i] = log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], sdt_lp[i]),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }
  {
    vector[N_false] B_trial = B_full[cell_false];
    vector[N_false] own_v   = vf[cell_false];
    vector[N_false] oth_v   = vk[cell_false];
    vector[N_false] own_sig = sigma_f[cell_false];
    vector[N_false] oth_sig = sigma_k[cell_false];
    vector[N_false] lapse_lp = rep_vector(-log(max_rt), N_false);
    vector[N_false] t_raw  = rt_false - t0;
    vector[N_false] t_safe = fmax(t_raw, 1e-4);
    vector[N_false] f_win =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));
    vector[N_false] F_lose =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));
    vector[N_false] sdt_lp = log(f_win) + log1m(F_lose);
    for (i in 1:N_false) {
      real log_w   = log_inv_logit(t_raw[i] / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw[i] / t0_blend_width);
      log_lik[N_correct + i] = log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], sdt_lp[i]),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }
}
