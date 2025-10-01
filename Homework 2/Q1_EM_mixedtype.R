# Q1: EM Algorithm for Mixed-Type Data
mixedtype_EM <- function(Y, K, max.iter, tol, mu.init, lambda.init, kappa.init, pi.init, quiet = TRUE) {
  if (ncol(Y) != 3) stop("Y must be an n x 3 matrix.")
  if (length(mu.init) != K || length(lambda.init) != K || 
      length(kappa.init) != K || length(pi.init) != K) stop("Initial values must have length K.")   

  n <- nrow(Y)
  mu <- mu.init
  lambda <- lambda.init
  kappa <- kappa.init
  pi <- pi.init
  loglik <- numeric(max.iter)

  logsumexp <- function(x) {
    max_x <- max(x)
    max_x + log(sum(exp(x - max_x)))
  }

  compute_Obsll <- function(Y, pi, lambda, kappa, mu) {
    n <- nrow(Y)
    K <- length(pi)
    ll_terms <- sapply(1:n, function(i) {
      logsumexp(sapply(1:K, function(k) {
        log(pi[k]) + 
          dpois(Y[i, 1], lambda[k], log = TRUE) + 
          dbinom(Y[i, 2], 1, kappa[k], log = TRUE) + 
          dnorm(Y[i, 3], mu[k], 1, log = TRUE)
      }))
    })
    sum(ll_terms)
  }

  for (iter in 1:max.iter) {
    gamma <- matrix(0, n, K)
    for (i in 1:n) {
      for (k in 1:K) {
        gamma[i, k] <- log(pi[k]) + 
                       dpois(Y[i, 1], lambda[k], log = TRUE) + 
                       dbinom(Y[i, 2], 1, kappa[k], log = TRUE) + 
                       dnorm(Y[i, 3], mu[k], 1, log = TRUE)
      }
      gamma[i, ] <- exp(gamma[i, ] - logsumexp(gamma[i, ])) 
    } 

    loglik[iter] <- compute_Obsll(Y, pi, lambda, kappa, mu)
    if (!quiet) cat("Iteration", iter, ": Log-likelihood =", loglik[iter], "\n")
    if (iter > 1 && abs((loglik[iter] - loglik[iter - 1]) / loglik[iter - 1]) < tol) {
      loglik <- loglik[1:iter]
      break
    }

    n_k <- colSums(gamma)
    pi <- n_k / n
    lambda <- sapply(1:K, function(k) sum(gamma[, k] * Y[, 1]) / n_k[k])
    kappa <- sapply(1:K, function(k) sum(gamma[, k] * Y[, 2]) / n_k[k])
    mu <- sapply(1:K, function(k) sum(gamma[, k] * Y[, 3]) / n_k[k])
  }

  list(mu = mu, lambda = lambda, kappa = kappa, pi = pi, loglik = loglik)
}

# Example usage (simulation)
set.seed(8054)
n <- 650; K <- 3
pi_true <- c(0.3, 0.4, 0.3)
lambda_true <- c(1, 2, 3)
kappa_true <- c(0.2, 0.5, 0.8)
mu_true <- c(-1, 0, 1)
Z <- sample(1:K, n, replace = TRUE, prob = pi_true)
Y <- matrix(0, nrow=n, ncol=3)
for (k in 1:n) {
  Y[k, 1] <- rpois(1, lambda_true[Z[k]])
  Y[k, 2] <- rbinom(1, 1, kappa_true[Z[k]])
  Y[k, 3] <- rnorm(1, mu_true[Z[k]], 1)
}

mu.init <- c(-0.5, 0, 0.5)
lambda.init <- c(0.5, 1.5, 2.5)
kappa.init <- c(0.1, 0.4, 0.7)
pi.init <- rep(1/3, 3)
max.iter <- 5000; tol <- 1e-8

out <- mixedtype_EM(Y, K, max.iter, tol, mu.init, lambda.init, kappa.init, pi.init)
str(out)
