data {
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;
  array[N_correct] int<lower=1,upper=3> cond_correct;
  array[N_false] int<lower=1,upper=3> cond_false;
  vector<lower=0>[3] max_rt;
  real<lower=0> t0_hi;   // t0 is scalar here - pool across all 3 conditions
}

transformed data {
  real t0_blend_width = 0.03;
}

parameters {
  vector<lower=0>[3] d;
  vector<lower=0>[3] B;

  real c;
  real<lower=0, upper=t0_hi> t0;
  real<lower=0.05, upper=50> r;
  real<lower=0,upper=1> p_lapse;
}

transformed parameters {
  vector[3] pk = fmin(fmax(Phi(d/2 - c), 1e-6), 1 - 1e-6);
  vector[3] pf = fmin(fmax(Phi(-d/2 - c), 1e-6), 1 - 1e-6);
  vector[3] vk = pk * r;
  vector[3] vf = pf * r;
  vector[3] sigma_k = fmax(sqrt(pk .* (1 - pk) * r), 0.05);
  vector[3] sigma_f = fmax(sqrt(pf .* (1 - pf) * r), 0.05);
}

model {
  t0 ~ normal(0.2, 0.15);
  d ~ normal(1.5, 0.5);
  c ~ normal(0, 0.5);
  B ~ exponential(1);
  r ~ gamma(9, 2);
  p_lapse ~ beta(1, 50);

  {
    vector[N_correct] B_trial = B[cond_correct];
    vector[N_correct] vk_trial = vk[cond_correct];
    vector[N_correct] vf_trial = vf[cond_correct];
    vector[N_correct] sigma_k_trial = sigma_k[cond_correct];
    vector[N_correct] sigma_f_trial = sigma_f[cond_correct];
    vector[N_correct] lapse_lp = -log(max_rt[cond_correct]);

    vector[N_correct] t_raw = rt_correct - t0;
    vector[N_correct] t_safe = fmax(t_raw, 1e-6);

    vector[N_correct] f_correct =
      B_trial ./ (sigma_k_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vk_trial .* t_safe).^2 ./ (2*sigma_k_trial.^2 .* t_safe));

    vector[N_correct] F_false =
      Phi((vf_trial .* t_safe - B_trial) ./ (sigma_f_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vf_trial) ./ sigma_f_trial.^2, 700))
      .* Phi(-(vf_trial .* t_safe + B_trial) ./ (sigma_f_trial .* sqrt(t_safe)));

    vector[N_correct] sdt_lp = log(f_correct) + log1m(F_false);

    for (i in 1:N_correct) {
      real log_w   = log_inv_logit(t_raw[i] / t0_blend_width);
      real log_1mw = log_inv_logit(-t_raw[i] / t0_blend_width);
      target += log_sum_exp(
        log_w   + log_mix(p_lapse, lapse_lp[i], sdt_lp[i]),
        log_1mw + log(p_lapse) + lapse_lp[i]
      );
    }
  }

  {
    vector[N_false] B_trial = B[cond_false];
    vector[N_false] vk_trial = vk[cond_false];
    vector[N_false] vf_trial = vf[cond_false];
    vector[N_false] sigma_k_trial = sigma_k[cond_false];
    vector[N_false] sigma_f_trial = sigma_f[cond_false];
    vector[N_false] lapse_lp = -log(max_rt[cond_false]);

    vector[N_false] t_raw = rt_false - t0;
    vector[N_false] t_safe = fmax(t_raw, 1e-6);

    vector[N_false] f_false =
      B_trial ./ (sigma_f_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vf_trial .* t_safe).^2 ./ (2*sigma_f_trial.^2 .* t_safe));

    vector[N_false] F_correct =
      Phi((vk_trial .* t_safe - B_trial) ./ (sigma_k_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vk_trial) ./ sigma_k_trial.^2, 700))
      .* Phi(-(vk_trial .* t_safe + B_trial) ./ (sigma_k_trial .* sqrt(t_safe)));

    vector[N_false] sdt_lp = log(f_false) + log1m(F_correct);

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
    vector[N_correct] B_trial = B[cond_correct];
    vector[N_correct] vk_trial = vk[cond_correct];
    vector[N_correct] vf_trial = vf[cond_correct];
    vector[N_correct] sigma_k_trial = sigma_k[cond_correct];
    vector[N_correct] sigma_f_trial = sigma_f[cond_correct];
    vector[N_correct] lapse_lp = -log(max_rt[cond_correct]);
    vector[N_correct] t_raw = rt_correct - t0;
    vector[N_correct] t_safe = fmax(t_raw, 1e-6);
    vector[N_correct] f_correct =
      B_trial ./ (sigma_k_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vk_trial .* t_safe).^2 ./ (2*sigma_k_trial.^2 .* t_safe));
    vector[N_correct] F_false =
      Phi((vf_trial .* t_safe - B_trial) ./ (sigma_f_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vf_trial) ./ sigma_f_trial.^2, 700))
      .* Phi(-(vf_trial .* t_safe + B_trial) ./ (sigma_f_trial .* sqrt(t_safe)));
    vector[N_correct] sdt_lp = log(f_correct) + log1m(F_false);
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
    vector[N_false] B_trial = B[cond_false];
    vector[N_false] vk_trial = vk[cond_false];
    vector[N_false] vf_trial = vf[cond_false];
    vector[N_false] sigma_k_trial = sigma_k[cond_false];
    vector[N_false] sigma_f_trial = sigma_f[cond_false];
    vector[N_false] lapse_lp = -log(max_rt[cond_false]);
    vector[N_false] t_raw = rt_false - t0;
    vector[N_false] t_safe = fmax(t_raw, 1e-6);
    vector[N_false] f_false =
      B_trial ./ (sigma_f_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vf_trial .* t_safe).^2 ./ (2*sigma_f_trial.^2 .* t_safe));
    vector[N_false] F_correct =
      Phi((vk_trial .* t_safe - B_trial) ./ (sigma_k_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vk_trial) ./ sigma_k_trial.^2, 700))
      .* Phi(-(vk_trial .* t_safe + B_trial) ./ (sigma_k_trial .* sqrt(t_safe)));
    vector[N_false] sdt_lp = log(f_false) + log1m(F_correct);
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
