data {
  int N_correct;
  int N_false;
  int P; // number of participants

  vector[N_correct] rt_correct;
  vector[N_false] rt_false;

  // combined participant x SAT index, values in 1..(3*P)
  array[N_correct] int<lower=1,upper=3*P> pc_correct;
  array[N_false] int<lower=1,upper=3*P> pc_false;

  // slowest observed RT for each participant x condition cell -
  // sets the range of the uniform lapse/guess component
  vector<lower=0>[3*P] max_rt;
  vector<lower=0>[3*P] t0_hi; // 5th-percentile RT per cell - keeps t0 away from the t=0 singularity
}


transformed data {
  // maps each flattened participant-condition slot back to its participant id
  array[3*P] int participant_of;
  for (p in 1:P) {
    for (k in 1:3) {
      participant_of[(p-1)*3 + k] = p;
    }
  }

  // per-trial participant id, gathered once (cheap, runs before sampling starts)
  array[N_correct] int pid_correct = participant_of[pc_correct];
  array[N_false] int pid_false = participant_of[pc_false];
}


parameters {
  vector[3*P] c;

  // t0 is now a free parameter - NOT bounded by any observed RT.
  // trials faster than t0 are handled explicitly below as guaranteed lapses,
  // rather than being made structurally impossible.
  vector<lower=0, upper=t0_hi>[3*P] t0;

  vector<lower=0>[P] d;
  vector<lower=0.05>[P] r;
  vector<lower=0>[P] B;

  // per-participant lapse/guess rate
  vector<lower=0,upper=1>[P] p_lapse;
}


transformed parameters {

  vector[3*P] d_rep = d[participant_of];
  vector[3*P] r_rep = r[participant_of];
  vector[3*P] B_rep = B[participant_of];

  vector[3*P] pk = fmin(fmax(Phi(-d_rep/2 - c), 1e-3), 1 - 1e-3);
  vector[3*P] pf = fmin(fmax(Phi(d_rep/2 - c), 1e-3), 1 - 1e-3);

  vector[3*P] vk = pk .* r_rep;
  vector[3*P] vf = pf .* r_rep;

  vector[3*P] sigma_k = sqrt(pk .* (1 - pk) .* r_rep);
  vector[3*P] sigma_f = sqrt(pf .* (1 - pf) .* r_rep);
}


model {

  // priors (independent across participants and conditions - no pooling)

  t0 ~ normal(0.2, 0.15); // plausible non-decision-time range; truncated at 0 by the constraint

  d ~ normal(1,1);

  c ~ normal(0,0.5);

  B ~ exponential(2);

  r ~ gamma(2, 5);

  p_lapse ~ beta(1, 100); // favors small lapse rates unless the data says otherwise



  // -----------------------
  // Correct trials
  // -----------------------

  {
    vector[N_correct] t0_trial = t0[pc_correct];
    vector[N_correct] vk_trial = vk[pc_correct];
    vector[N_correct] vf_trial = vf[pc_correct];
    vector[N_correct] sigma_k_trial = sigma_k[pc_correct];
    vector[N_correct] sigma_f_trial = sigma_f[pc_correct];
    vector[N_correct] B_trial = B_rep[pc_correct];

    vector[N_correct] p_lapse_trial = p_lapse[pid_correct];
    vector[N_correct] lapse_lp = -log(max_rt[pc_correct]); // uniform(0, max_rt) log-density

    vector[N_correct] t_raw;
    vector[N_correct] t_safe;
    vector[N_correct] f_correct;
    vector[N_correct] F_false;
    vector[N_correct] sdt_lp;


    t_raw = rt_correct - t0_trial;

    // clamp only so the vectorized math below stays well-defined;
    // the ORIGINAL t_raw (not this clamped copy) decides lapse-vs-diffusion below
    t_safe = fmax(t_raw, 1e-6);


    f_correct =
      B_trial ./ (sigma_k_trial .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B_trial - vk_trial .* t_safe).^2 ./ (2*sigma_k_trial.^2 .* t_safe));


    F_false =
      Phi((vf_trial .* t_safe - B_trial) ./ (sigma_f_trial .* sqrt(t_safe)))
      + exp((2*B_trial .* vf_trial) ./ sigma_f_trial.^2)
      .* Phi(-(vf_trial .* t_safe + B_trial) ./ (sigma_f_trial .* sqrt(t_safe)));


    sdt_lp = log(f_correct) + log1m(F_false);

    // log_mix has no vectorized form, and this trial-by-trial branch (real
    // decision vs. structurally-impossible-therefore-lapse) needs a loop regardless
    for (i in 1:N_correct) {
      if (t_raw[i] > 0) {
        target += log_mix(p_lapse_trial[i], lapse_lp[i], sdt_lp[i]);
      } else {
        // negative decision time: the diffusion account is impossible here,
        // so this trial MUST be a lapse
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
    vector[N_false] B_trial = B_rep[pc_false];

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
      + exp((2*B_trial .* vk_trial) ./ sigma_k_trial.^2)
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

