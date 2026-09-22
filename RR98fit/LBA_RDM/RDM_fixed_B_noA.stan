// "RDM_free, no A" -- plain Wald race (NO starting-point variability),
// matching SRDM's own race architecture exactly. B fixed externally
// (from SRDM), v_correct/v_error/s_correct/s_error free per (difficulty
// x instruction) cell, same as RDM_fixed_B_by_instruction.stan.
//
// WHY THIS VERSION: SRDM has no starting-point variability (A=0,
// implicitly) -- it's a plain Wald race. RDM_free's A>0 gives it an
// extra mechanism for producing fast errors that SRDM structurally
// can't use, which could easily explain the systematic v/s divergence
// seen when comparing the two. Rather than take a numerically risky
// A->0 limit of the with-A formula (dividing two vanishing quantities),
// this uses the plain closed-form Wald pdf/cdf directly -- the exact
// same formulas already validated and used in the SRDM Stan file itself
// (f_win / F_lose), just with v/s freely estimated per cell instead of
// derived from d'/c/r. This makes the comparison to SRDM as close to
// apples-to-apples as the two model classes allow: same race
// architecture, same absence of starting-point variability, differing
// only in how v/s per cell are parameterized.

data {
  int<lower=1> N_LEVELS;
  int N_correct;
  int N_false;
  vector[N_correct] rt_correct;
  vector[N_false] rt_false;
  array[N_correct] int<lower=1,upper=2*N_LEVELS> cell_correct;
  array[N_false]   int<lower=1,upper=2*N_LEVELS> cell_false;
  real<lower=0> t0_hi;
  real<lower=0> B;   // FIXED -- set this to SRDM's own fitted boundary value
}

transformed data {
  int N_CELLS = 2 * N_LEVELS;
}

parameters {
  vector<lower=0>[N_CELLS] v_correct;
  vector<lower=0>[N_CELLS] v_error;
  vector<lower=0>[N_CELLS] s_correct;
  vector<lower=0>[N_CELLS] s_error;
  real<lower=0, upper=t0_hi> t0;
}

model {
  v_correct ~ normal(2.5, 1.5) T[0,];
  v_error ~ normal(1.0, 1.0) T[0,];
  s_correct ~ normal(1.0, 1.0) T[0,];
  s_error ~ normal(1.0, 1.0) T[0,];
  t0 ~ normal(0.25, 0.05);

  // Correct trials: own accumulator wins, other accumulator loses
  {
    vector[N_correct] vc = v_correct[cell_correct];
    vector[N_correct] ve = v_error[cell_correct];
    vector[N_correct] sc = s_correct[cell_correct];
    vector[N_correct] se = s_error[cell_correct];
    vector[N_correct] t_safe = fmax(rt_correct - t0, 1e-4);

    vector[N_correct] f_win = B ./ (sc .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B - vc .* t_safe).^2 ./ (2*sc.^2 .* t_safe));
    vector[N_correct] F_lose = Phi((ve .* t_safe - B) ./ (se .* sqrt(t_safe)))
      + exp(fmin((2*B .* ve) ./ se.^2, 700))
      .* Phi(-(ve .* t_safe + B) ./ (se .* sqrt(t_safe)));

    target += sum(log(f_win) + log1m(F_lose));
  }

  // Error trials: mirror roles
  {
    vector[N_false] vc = v_correct[cell_false];
    vector[N_false] ve = v_error[cell_false];
    vector[N_false] sc = s_correct[cell_false];
    vector[N_false] se = s_error[cell_false];
    vector[N_false] t_safe = fmax(rt_false - t0, 1e-4);

    vector[N_false] f_win = B ./ (se .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B - ve .* t_safe).^2 ./ (2*se.^2 .* t_safe));
    vector[N_false] F_lose = Phi((vc .* t_safe - B) ./ (sc .* sqrt(t_safe)))
      + exp(fmin((2*B .* vc) ./ sc.^2, 700))
      .* Phi(-(vc .* t_safe + B) ./ (sc .* sqrt(t_safe)));

    target += sum(log(f_win) + log1m(F_lose));
  }
}

generated quantities {
  vector[N_correct + N_false] log_lik;
  {
    vector[N_correct] vc = v_correct[cell_correct];
    vector[N_correct] ve = v_error[cell_correct];
    vector[N_correct] sc = s_correct[cell_correct];
    vector[N_correct] se = s_error[cell_correct];
    vector[N_correct] t_safe = fmax(rt_correct - t0, 1e-4);
    vector[N_correct] f_win = B ./ (sc .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B - vc .* t_safe).^2 ./ (2*sc.^2 .* t_safe));
    vector[N_correct] F_lose = Phi((ve .* t_safe - B) ./ (se .* sqrt(t_safe)))
      + exp(fmin((2*B .* ve) ./ se.^2, 700))
      .* Phi(-(ve .* t_safe + B) ./ (se .* sqrt(t_safe)));
    for (i in 1:N_correct)
      log_lik[i] = log(f_win[i]) + log1m(F_lose[i]);
  }
  {
    vector[N_false] vc = v_correct[cell_false];
    vector[N_false] ve = v_error[cell_false];
    vector[N_false] sc = s_correct[cell_false];
    vector[N_false] se = s_error[cell_false];
    vector[N_false] t_safe = fmax(rt_false - t0, 1e-4);
    vector[N_false] f_win = B ./ (se .* sqrt(2*pi()*pow(t_safe,3)))
      .* exp(-(B - ve .* t_safe).^2 ./ (2*se.^2 .* t_safe));
    vector[N_false] F_lose = Phi((vc .* t_safe - B) ./ (sc .* sqrt(t_safe)))
      + exp(fmin((2*B .* vc) ./ sc.^2, 700))
      .* Phi(-(vc .* t_safe + B) ./ (sc .* sqrt(t_safe)));
    for (i in 1:N_false)
      log_lik[N_correct + i] = log(f_win[i]) + log1m(F_lose[i]);
  }
}
