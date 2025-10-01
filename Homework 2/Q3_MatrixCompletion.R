# Q3: Matrix Completion with Nuclear Norm Penalty
library(Matrix); library(foreach); library(doParallel)

prox_nuclear <- function(X, lambda) {
  svd_decomp <- svd(X)
  s_thresholded <- pmax(svd_decomp$d - lambda, 0)
  svd_decomp$u %*% diag(s_thresholded) %*% t(svd_decomp$v)
}

compute_lambda_max <- function(Y, Omega) {
  Y_proj <- Y * Omega
  norm(Y_proj, type = "2") / sum(Omega)
}

objective_function <- function(Y, Omega, Theta, lambda) {
  residual <- (Y - Theta) * Omega
  0.5 * sum(residual^2) / sum(Omega) + lambda * sum(svd(Theta)$d)
}

grad_g <- function(Y, Omega, Theta) {
  -((Y - Theta) * Omega) / sum(Omega)
}

backtracking_line_search <- function(Y, Omega, Theta, grad, lambda, initial_step, beta = 0.5, c = 0.5, max_iter = 1000) {
  step_size <- initial_step
  obj_current <- objective_function(Y, Omega, Theta, lambda)
  for (i in 1:max_iter) {
    Theta_new <- prox_nuclear(Theta - step_size * grad, lambda * step_size)
    obj_new <- objective_function(Y, Omega, Theta_new, lambda)
    if (obj_new <= obj_current - c * step_size * sum(grad^2)) break
    step_size <- step_size * beta
  }
  step_size
}

prox_grad_descent_single_lambda <- function(Y, Omega, lambda, max_iter = 20000, tol = 1e-8, Theta_init = NULL, initial_step) {
  n <- nrow(Y); m <- ncol(Y)
  Theta <- if (!is.null(Theta_init)) Theta_init else matrix(0, n, m)
  prev_obj <- Inf; consecutive_small_changes <- 0

  for (iteration in 1:max_iter) {
    grad <- grad_g(Y, Omega, Theta)
    step_size <- backtracking_line_search(Y, Omega, Theta, grad, lambda, initial_step)
    Theta_new <- prox_nuclear(Theta - step_size * grad, lambda * step_size)
    obj <- objective_function(Y, Omega, Theta_new, lambda)

    relative_change <- if (is.infinite(prev_obj)) Inf else abs(prev_obj - obj) / (abs(prev_obj) + 1e-8)

    if (relative_change < tol) consecutive_small_changes <- consecutive_small_changes + 1 else consecutive_small_changes <- 0
    if (consecutive_small_changes >= 3) break

    Theta <- Theta_new; prev_obj <- obj
  }
  Theta
}

matrix_completion_path <- function(Y, nlambda, lambda_ratio, max_iter = 20000, tol = 1e-8, initial_step) {
  Omega <- !is.na(Y); Y_obs <- Y; Y_obs[!Omega] <- 0
  lambda_max <- compute_lambda_max(Y_obs, Omega)
  lambda_vec <- 10^seq(log10(lambda_max), log10(lambda_ratio*lambda_max), length.out = nlambda)
  Theta_list <- list(); Theta_init <- NULL
  for (lambda in lambda_vec) {
    Theta <- prox_grad_descent_single_lambda(Y_obs, Omega, lambda, max_iter, tol, Theta_init, initial_step)
    Theta_list <- append(Theta_list, list(Theta))
    Theta_init <- Theta
  }
  list(Theta.list = Theta_list, lambda.vec = lambda_vec)
}

matrix_completion_C
