data {
  int N_correct;
  int N_false;
  int P; // number of participants

  vector[N_correct] rt_correct;
  vector[N_false] rt_false;

  // combined participant x SAT index, values in 1..(3*P)
  array[N_correct] int<lower=1,upper=3*P> pc_correct;
  array[N_false] int<lower=1,upper=3*P> pc_false;

  vector<lower=0>[3*P] max_rt;  // uniform lapse range, per cell
  vector<lower=0>[3*P] t0_hi;   // 5th-percentile RT per cell, upper bound for t0
}


transformed data {
  // which of the 3 conditions each flattened participant-condition slot belongs to
  array[3*P] int cond_of;
  // which participant each flattened participant-condition slot belongs to
  array[3*P] int participant_of;
  for (p in 1:P) {
    for (k in 1:3) {
      cond_of[(p-1)*3 + k] = k;
      participant_of[(p-1)*3 + k] = p;
    }
  }

  // per-trial participant id, gathered once (cheap, runs before sampling starts)
  array[N_correct] int pid_correct = participant_of[pc_correct];
  array[N_false] int pid_false = participant_of[pc_false];
}


parameters {
  // ---- population-level (hyper) parameters, one value per SAT condition ----
  vector[3] mu_c;
  vector<lower=0>[3] sigma_c;

  vector[3] mu_logB;
  vector<lower=0>[3] sigma_logB;

  vector[3] mu_t0_logit;
  vector<lower=0>[3] sigma_t0_logit;

  // ---- collapsed: single population value, shared across conditions ----
  real mu_logd;
  real<lower=0> sigma_logd;

  real mu_logr;
  real<lower=0> sigma_logr;

  real mu_lapse_logit;
  real<lower=0> sigma_lapse_logit;

  // ---- non-centered per participant x condition deviations ----
  vector[3*P] c_z;
  vector[3*P] logB_z;
  vector[3*P] t0_z;

  // ---- non-centered per-participant-only deviations (collapsed parameters) ----
  vector[P] logd_z;
  vector[P] logr_z;
  vector[P] lapse_z;
}


transformed parameters {

  // partially-pooled per-cell parameters, built via non-centered transforms
  vector[3*P] c  = mu_c[cond_of] + sigma_c[cond_of] .* c_z;
  vector[3*P] B  = exp(mu_logB[cond_of] + sigma_logB[cond_of] .* logB_z);
  vector[3*P] t0 = t0_hi .* inv_logit(mu_t0_logit[cond_of] + sigma_t0_logit[cond_of] .* t0_z);

  // collapsed: one value per participant, shared across their 3 conditions
  vector[P] d = exp(mu_logd + sigma_logd * logd_z);
  vector[P] r = exp(mu_logr + sigma_logr * logr_z);
  vector[P] p_lapse = inv_logit(mu_lapse_logit + sigma_lapse_logit * lapse_z);

  // gathered up to per-cell length wherever needed alongside c/B
  vector[3*P] d_rep = d[participant_of];
  vector[3*P] r_rep = r[participant_of];

  vector[3*P] pk = fmin(fmax(Phi(d_rep/2 - c), 1e-6), 1 - 1e-6);
  vector[3*P] pf = fmin(fmax(Phi(-d_rep/2 - c), 1e-6), 1 - 1e-6);

  vector[3*P] vk = pk .* r_rep;
  vector[3*P] vf = pf .* r_rep;

  vector[3*P] sigma_k = fmax(sqrt(pk .* (1 - pk) .* r_rep), 0.05);
  vector[3*P] sigma_f = fmax(sqrt(pf .* (1 - pf) .* r_rep), 0.05);
}


model {

  // ---- hyperpriors ----
  mu_c ~ normal(0, 1);
  sigma_c ~ exponential(2);

  mu_logB ~ normal(log(1.0), 0.5);   // reference doc: B ~0.8-1.7 non-degenerate
  sigma_logB ~ exponential(3);

  mu_t0_logit ~ normal(0, 1);        // centered on ~50% of each cell's t0_hi
  sigma_t0_logit ~ exponential(2);

  // collapsed hyperpriors - single population value now
  mu_logd ~ normal(log(1.5), 0.5);   // reference doc: d' mean ~1.5
  sigma_logd ~ exponential(3);

  mu_logr ~ normal(log(4.5), 0.5);   // reference doc: r mean ~4.5
  sigma_logr ~ exponential(3);

  mu_lapse_logit ~ normal(logit(0.02), 1); // centered on a small ~2% lapse rate
  sigma_lapse_logit ~ exponential(2);

  // ---- non-centered raw deviations ----
  c_z ~ std_normal();
  logB_z ~ std_normal();
  t0_z ~ std_normal();
  logd_z ~ std_normal();
  logr_z ~ std_normal();
  lapse_z ~ std_normal();



  // -----------------------
  // Correct trials
  // -----------------------

  {
    vector[N_correct] t0_trial = t0[pc_correct];
    vector[N_correct] vk_trial = vk[pc_correct];
    vector[N_correct] vf_trial = vf[pc_correct];
    vector[N_correct] sigma_k_trial = sigma_k[pc_correct];
    vector[N_correct] sigma_f_trial = sigma_f[pc_correct];
    vector[N_correct] B_trial = B[pc_correct];

    vector[N_correct] p_lapse_trial = p_lapse[pid_correct];
    vector[N_correct] lapse_lp = -log(max_rt[pc_correct]);

    vector[N_correct] t_raw;
    vector[N_correct] t_safe;
    vector[N_correct] f_correct;
    vector[N_correct] F_false;
    vector[N_correct] sdt_lp;


    t_raw = rt_correct - t0_trial;
    t_safe = fmax(t_raw, 1e-6);


    f_correct =
      B_trial ./ (sigma_k_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vk_trial .* t_safe).^2 ./ (2*sigma_k_trial.^2 .* t_safe));


    F_false =
      Phi((vf_trial .* t_safe - B_trial) ./ (sigma_f_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vf_trial) ./ sigma_f_trial.^2, 700))
      .* Phi(-(vf_trial .* t_safe + B_trial) ./ (sigma_f_trial .* sqrt(t_safe)));


    sdt_lp = log(f_correct) + log1m(F_false);

    for (i in 1:N_correct) {
      if (t_raw[i] > 0) {
        target += log_mix(p_lapse_trial[i], lapse_lp[i], sdt_lp[i]);
      } else {
        target += log(p_lapse_trial[i]) + lapse_lp[i];
      }
    }
  }



  // -----------------------
  // False trials
  // -----------------------

  {
    vector[N_false] t0_trial = t0[pc_false];
    vector[N_false] vk_trial = vk[pc_false];
    vector[N_false] vf_trial = vf[pc_false];
    vector[N_false] sigma_k_trial = sigma_k[pc_false];
    vector[N_false] sigma_f_trial = sigma_f[pc_false];
    vector[N_false] B_trial = B[pc_false];

    vector[N_false] p_lapse_trial = p_lapse[pid_false];
    vector[N_false] lapse_lp = -log(max_rt[pc_false]);

    vector[N_false] t_raw;
    vector[N_false] t_safe;
    vector[N_false] f_false;
    vector[N_false] F_correct;
    vector[N_false] sdt_lp;


    t_raw = rt_false - t0_trial;
    t_safe = fmax(t_raw, 1e-6);


    f_false =
      B_trial ./ (sigma_f_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vf_trial .* t_safe).^2 ./ (2*sigma_f_trial.^2 .* t_safe));


    F_correct =
      Phi((vk_trial .* t_safe - B_trial) ./ (sigma_k_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vk_trial) ./ sigma_k_trial.^2, 700))
      .* Phi(-(vk_trial .* t_safe + B_trial) ./ (sigma_k_trial .* sqrt(t_safe)));


    sdt_lp = log(f_false) + log1m(F_correct);

    for (i in 1:N_false) {
      if (t_raw[i] > 0) {
        target += log_mix(p_lapse_trial[i], lapse_lp[i], sdt_lp[i]);
      } else {
        target += log(p_lapse_trial[i]) + lapse_lp[i];
      }
    }
  }

}


generated quantities {
  // pointwise log-likelihood, one entry per trial, for LOO/WAIC comparison
  // (az.from_cmdstanpy(fit, log_likelihood="log_lik") reads this directly)
  vector[N_correct + N_false] log_lik;

  {
    vector[N_correct] t0_trial = t0[pc_correct];
    vector[N_correct] vk_trial = vk[pc_correct];
    vector[N_correct] vf_trial = vf[pc_correct];
    vector[N_correct] sigma_k_trial = sigma_k[pc_correct];
    vector[N_correct] sigma_f_trial = sigma_f[pc_correct];
    vector[N_correct] B_trial = B[pc_correct];

    vector[N_correct] p_lapse_trial = p_lapse[pid_correct];
    vector[N_correct] lapse_lp = -log(max_rt[pc_correct]);

    vector[N_correct] t_raw;
    vector[N_correct] t_safe;
    vector[N_correct] f_correct;
    vector[N_correct] F_false;
    vector[N_correct] sdt_lp;

    t_raw = rt_correct - t0_trial;
    t_safe = fmax(t_raw, 1e-6);

    f_correct =
      B_trial ./ (sigma_k_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vk_trial .* t_safe).^2 ./ (2*sigma_k_trial.^2 .* t_safe));

    F_false =
      Phi((vf_trial .* t_safe - B_trial) ./ (sigma_f_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vf_trial) ./ sigma_f_trial.^2, 700))
      .* Phi(-(vf_trial .* t_safe + B_trial) ./ (sigma_f_trial .* sqrt(t_safe)));

    sdt_lp = log(f_correct) + log1m(F_false);

    for (i in 1:N_correct) {
      if (t_raw[i] > 0) {
        log_lik[i] = log_mix(p_lapse_trial[i], lapse_lp[i], sdt_lp[i]);
      } else {
        log_lik[i] = log(p_lapse_trial[i]) + lapse_lp[i];
      }
    }
  }

  {
    vector[N_false] t0_trial = t0[pc_false];
    vector[N_false] vk_trial = vk[pc_false];
    vector[N_false] vf_trial = vf[pc_false];
    vector[N_false] sigma_k_trial = sigma_k[pc_false];
    vector[N_false] sigma_f_trial = sigma_f[pc_false];
    vector[N_false] B_trial = B[pc_false];

    vector[N_false] p_lapse_trial = p_lapse[pid_false];
    vector[N_false] lapse_lp = -log(max_rt[pc_false]);

    vector[N_false] t_raw;
    vector[N_false] t_safe;
    vector[N_false] f_false;
    vector[N_false] F_correct;
    vector[N_false] sdt_lp;

    t_raw = rt_false - t0_trial;
    t_safe = fmax(t_raw, 1e-6);

    f_false =
      B_trial ./ (sigma_f_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vf_trial .* t_safe).^2 ./ (2*sigma_f_trial.^2 .* t_safe));

    F_correct =
      Phi((vk_trial .* t_safe - B_trial) ./ (sigma_k_trial .* sqrt(t_safe)))
      + exp(fmin((2*B_trial .* vk_trial) ./ sigma_k_trial.^2, 700))
      .* Phi(-(vk_trial .* t_safe + B_trial) ./ (sigma_k_trial .* sqrt(t_safe)));

    sdt_lp = log(f_false) + log1m(F_correct);

    for (i in 1:N_false) {
      if (t_raw[i] > 0) {
        log_lik[N_correct + i] = log_mix(p_lapse_trial[i], lapse_lp[i], sdt_lp[i]);
      } else {
        log_lik[N_correct + i] = log(p_lapse_trial[i]) + lapse_lp[i];
      }
    }
  }
}
