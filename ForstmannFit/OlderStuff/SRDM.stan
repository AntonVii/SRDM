data {
  int N_correct;
  int N_false;
  int P; // number of participants

  vector[N_correct] rt_correct;
  vector[N_false] rt_false;

  // combined participant x SAT index, values in 1..(3*P)
  // build in R/Python as: (participant_id - 1) * 3 + sat_id
  array[N_correct] int<lower=1,upper=3*P> pc_correct;
  array[N_false] int<lower=1,upper=3*P> pc_false;
}


transformed data {
  // maps each flattened participant-condition slot back to its participant id
  array[3*P] int participant_of;
  for (p in 1:P) {
    for (k in 1:3) {
      participant_of[(p-1)*3 + k] = p;
    }
  }
}


parameters {
  vector[3*P] c;
  vector<lower=0,upper=0.25>[3*P] t0;

  vector[P] d;
  vector<lower=0>[P] r;
  vector<lower=0>[P] B;
}


transformed parameters {

  vector[3*P] d_rep = d[participant_of];
  vector[3*P] r_rep = r[participant_of];
  vector[3*P] B_rep = B[participant_of];

  vector[3*P] pk = Phi(-d_rep/2 - c);
  vector[3*P] pf = Phi(d_rep/2 - c);

  vector[3*P] vk = pk .* r_rep;
  vector[3*P] vf = pf .* r_rep;

  vector[3*P] sigma_k = sqrt(pk .* (1 - pk) .* r_rep);
  vector[3*P] sigma_f = sqrt(pf .* (1 - pf) .* r_rep);
}


model {

  // priors (independent across participants and conditions - no pooling)

  t0 ~ exponential(0.2);

  d ~ normal(0,3);

  c ~ normal(0,1);

  B ~ exponential(2);

  r ~ exponential(0.2);



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

    vector[N_correct] t;
    vector[N_correct] f_correct;
    vector[N_correct] F_false;


    t = rt_correct - t0_trial;


    f_correct =
      B_trial ./ (sigma_k_trial .* sqrt(2*pi()*pow(t,3)))
      .* exp(-(B_trial - vk_trial .* t).^2 ./ (2*sigma_k_trial.^2 .* t));


    F_false =
      Phi((vf_trial .* t - B_trial) ./ (sigma_f_trial .* sqrt(t)))
      + exp((2*B_trial .* vf_trial) ./ sigma_f_trial.^2)
      .* Phi(-(vf_trial .* t + B_trial) ./ (sigma_f_trial .* sqrt(t)));


    target += sum(log(f_correct .* (1 - F_false)));
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

    vector[N_false] t;
    vector[N_false] f_false;
    vector[N_false] F_correct;


    t = rt_false - t0_trial


    f_false =
      B_trial ./ (sigma_f_trial .* sqrt(2*pi()*pow(t,3)))
      .* exp(-(B_trial - vf_trial .* t).^2 ./ (2*sigma_f_trial.^2 .* t));


    F_correct =
      Phi((vk_trial .* t - B_trial) ./ (sigma_k_trial .* sqrt(t)))
      + exp((2*B_trial .* vk_trial) ./ sigma_k_trial.^2)
      .* Phi(-(vk_trial .* t + B_trial) ./ (sigma_k_trial .* sqrt(t)));


    target += sum(log(f_false .* (1 - F_correct)));
  }

}
