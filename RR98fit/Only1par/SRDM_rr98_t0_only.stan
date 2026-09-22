data {
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;
  array[N_correct] int<lower=1,upper=14> cell_correct;
  array[N_false] int<lower=1,upper=14> cell_false;
  real<lower=0> max_rt;
  vector<lower=0>[2] t0_hi;   // t0 varies by SAT - per-SAT bound
}

transformed data {
  real t0_blend_width = 0.03;
  array[14] int sat_of;
  array[14] int diff_of;
  for (s in 1:2) {
    for (l in 1:7) {
      int idx = (s-1)*7 + l;
      sat_of[idx] = s;
      diff_of[idx] = l;
    }
  }


  array[N_correct] int sat_id_correct = sat_of[cell_correct];
  array[N_false] int sat_id_false = sat_of[cell_false];
}

parameters {
  real c;
  real<lower=0> B;
  vector<lower=0,upper=t0_hi>[2] t0;
  vector<lower=0>[7] d_base; // difficulty-driven d' curve, shared across SAT
  real<lower=0.05, upper=50> r;
  real<lower=0,upper=1> p_lapse;
}

transformed parameters {
  vector[14] c_full = rep_vector(c, 14);
  vector[14] B_full = rep_vector(B, 14);
  vector[14] d_full = d_base[diff_of];

  vector[14] pk = fmin(fmax(Phi(d_full/2 - c_full), 1e-6), 1 - 1e-6);
  vector[14] pf = fmin(fmax(Phi(-d_full/2 - c_full), 1e-6), 1 - 1e-6);
  vector[14] vk = pk .* r;
  vector[14] vf = pf .* r;
  vector[14] sigma_k = fmax(sqrt(pk .* (1 - pk) * r), 0.05);
  vector[14] sigma_f = fmax(sqrt(pf .* (1 - pf) * r), 0.05);
}

model {
  t0 ~ normal(0.2, 0.15);
  d_base ~ normal(1.5, 0.5);
  c ~ normal(0, 0.5);
  B ~ exponential(1);
  r ~ gamma(9, 2);
  p_lapse ~ beta(1, 50);

  {
    vector[N_correct] B_trial = B_full[cell_correct];
    vector[N_correct] t0_trial_v = t0[sat_id_correct];
    vector[N_correct] own_v = vk[cell_correct];
    vector[N_correct] oth_v = vf[cell_correct];
    vector[N_correct] own_sig = sigma_k[cell_correct];
    vector[N_correct] oth_sig = sigma_f[cell_correct];
    vector[N_correct] lapse_lp = rep_vector(-log(max_rt), N_correct);

    vector[N_correct] t_raw = rt_correct - t0_trial_v;
    vector[N_correct] t_safe = fmax(t_raw, 1e-4);

    vector[N_correct] f_correct =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));

    vector[N_correct] F_false =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));

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
    vector[N_false] B_trial = B_full[cell_false];
    vector[N_false] t0_trial_v = t0[sat_id_false];
    vector[N_false] own_v = vf[cell_false];
    vector[N_false] oth_v = vk[cell_false];
    vector[N_false] own_sig = sigma_f[cell_false];
    vector[N_false] oth_sig = sigma_k[cell_false];
    vector[N_false] lapse_lp = rep_vector(-log(max_rt), N_false);

    vector[N_false] t_raw = rt_false - t0_trial_v;
    vector[N_false] t_safe = fmax(t_raw, 1e-4);

    vector[N_false] f_false =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));

    vector[N_false] F_correct =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));

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
    vector[N_correct] B_trial = B_full[cell_correct];
    vector[N_correct] t0_trial_v = t0[sat_id_correct];
    vector[N_correct] own_v = vk[cell_correct];
    vector[N_correct] oth_v = vf[cell_correct];
    vector[N_correct] own_sig = sigma_k[cell_correct];
    vector[N_correct] oth_sig = sigma_f[cell_correct];
    vector[N_correct] lapse_lp = rep_vector(-log(max_rt), N_correct);

    vector[N_correct] t_raw = rt_correct - t0_trial_v;
    vector[N_correct] t_safe = fmax(t_raw, 1e-4);

    vector[N_correct] f_correct =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));

    vector[N_correct] F_false =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));

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
    vector[N_false] B_trial = B_full[cell_false];
    vector[N_false] t0_trial_v = t0[sat_id_false];
    vector[N_false] own_v = vf[cell_false];
    vector[N_false] oth_v = vk[cell_false];
    vector[N_false] own_sig = sigma_f[cell_false];
    vector[N_false] oth_sig = sigma_k[cell_false];
    vector[N_false] lapse_lp = rep_vector(-log(max_rt), N_false);

    vector[N_false] t_raw = rt_false - t0_trial_v;
    vector[N_false] t_safe = fmax(t_raw, 1e-4);

    vector[N_false] f_false =
      B_trial ./ (own_sig .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - own_v .* t_safe).^2 ./ (2*own_sig.^2 .* t_safe));

    vector[N_false] F_correct =
      Phi((oth_v .* t_safe - B_trial) ./ (oth_sig .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* oth_v) ./ oth_sig.^2, 700))
      .* Phi(-(oth_v .* t_safe + B_trial) ./ (oth_sig .* sqrt(t_safe)));

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
