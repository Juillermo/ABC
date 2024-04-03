data {
  int<lower=1> n;  // Number of morbidities
  array[n,n] int<lower=0> X;  // Co-occurrence observation matrix
  int<lower=1> M;  // number of patients
  array[n] int<lower=0, upper=M> P;  // Prevalence of conditions
  
  // Upper bounds of parameters
  array[n,n] real<lower=0> upper_r;
  array[n] real<lower=0> upper_sigma;
}
parameters {
  array[n] real<lower=0, upper=upper_sigma> sigma;  // prob of a morbidity to emerge independently
  array[n,n] real<lower=-1, upper=upper_r> r;  // positive co-morbidity interactions
  
  // Lognormal prior for associations
  real mu_lognormal_prior;
  real<lower=0> std_lognormal_prior;
  
  // Beta prior for sigma values
  real<lower=1> alpha_beta_prior;
  real<lower=1> beta_beta_prior;
}
model {
  // Priors
  alpha_beta_prior - 1 ~ exponential(0.01);
  beta_beta_prior - 1 ~ exponential(0.01);
  for (i in 1:n) {
    sigma[i] ~ beta(alpha_beta_prior, beta_beta_prior);
  }
  
  mu_lognormal_prior ~ normal(0, 1);
  std_lognormal_prior ~ exponential(1);
  for (i in 1:n) {
    for (j in i + 1:n) {
      // r[i,j] + 1 ~ lognormal(mu_lognormal_prior, std_lognormal_prior);
      r[i,j] + 1 ~ lognormal(0, std_lognormal_prior);
      
      // Likelihood
      real mu_11 = sigma[i] * sigma[j] * (1 + r[i,j] * (1 - sigma[i] * sigma[j]));
      real mu_10 = sigma[i] * (1 - sigma[j]) * (1 - sigma[i] * sigma[j] * r[i,j]);
      real mu_01 = sigma[j] * (1 - sigma[i]) * (1 - sigma[j] * sigma[i] * r[i,j]);
      real mu_00 = (1 - sigma[i]) * (1 - sigma[j]) * (1 - sigma[i] * sigma[j] * r[i,j]);
      
      array[4] int y = { X[i,j], P[i] - X[i,j], P[j] - X[i,j], M - P[i] - P[j] + X[i,j] };
      vector[4] theta = [mu_11, mu_10, mu_01, mu_00]';
      y ~ multinomial(theta);
    }
  }
}
generated quantities {
  // Prior predictive check
  real<lower=1> alpha_prior = 1 + exponential_rng(0.01);
  real<lower=1> beta_prior = 1 + exponential_rng(0.01);
  array[n] real sigma_priors;
  array[n] real sigma_prior_post;
  for (i in 1:n) {
    sigma_priors[i] = beta_rng(alpha_prior, beta_prior);
    sigma_prior_post[i] = beta_rng(alpha_beta_prior, beta_beta_prior);
  }
  
  real mu_prior = normal_rng(0, 1);
  real<lower=0> std_prior = exponential_rng(1);
  array[n,n] real r_priors;
  array[n,n] real rate_X_prior;
  array[n,n] real rate_X_post;
  
  array[n,n] int X_post;
  array[n,n] int P_post;
  for (i in 1:n){
    for (j in i + 1:n){
      r_priors[i,j] = - 1 + lognormal_rng(mu_prior, std_prior);
  
      rate_X_prior[i,j] = sigma_priors[i] * sigma_priors[j] * (1 + r_priors[i,j] * (1 - sigma_priors[i] * sigma_priors[j]));

      real mu_11 = sigma[i] * sigma[j] * (1 + r[i,j] * (1 - sigma[i] * sigma[j]));
      real mu_10 = sigma[i] * (1 - sigma[j]) * (1 - sigma[i] * sigma[j] * r[i,j]);
      real mu_01 = sigma[j] * (1 - sigma[i]) * (1 - sigma[j] * sigma[i] * r[i,j]);
      real mu_00 = (1 - sigma[i]) * (1 - sigma[j]) * (1 - sigma[i] * sigma[j] * r[i,j]);
      vector[4] theta = [mu_11, mu_10, mu_01, mu_00]';
      
      array[4] int y = multinomial_rng(theta, M);
      X_post[i,j] = y[1];
      P_post[i,j] = y[2] + y[1];
      P_post[j,i] = y[3] + y[1];
    }
  }
}