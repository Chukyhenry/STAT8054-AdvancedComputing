library(MASS)

QNewton_MultinomLogisticReg <- function(X, y, max_iter = 1000, tol = 1e-8) {
  X <- as.matrix(X)
  y <- as.integer(y)
  n <- nrow(X)
  p <- ncol(X)
  K <- max(y)
  stopifnot(K >= 2, all(y >= 1 & y <= K))
  
  beta_init <- rep(0, (K-1)*p)
  obj_values <- numeric(0)
  
  neg_loglik <- function(beta_flat) {
    beta_mat <- matrix(beta_flat, nrow = K-1, ncol = p)
    scores <- X %*% t(beta_mat)
    m <- pmax(0, apply(scores, 1, max))
    sum_exp <- rowSums(exp(scores - m)) + exp(-m)
    log_sum <- m + log(sum_exp)
    log_probs <- numeric(n)
    is_ref <- (y == K)
    log_probs[is_ref] <- -log_sum[is_ref]
    non_ref <- which(!is_ref)
    y_non_ref <- y[non_ref]
    log_probs[non_ref] <- scores[cbind(non_ref, y_non_ref)] - log_sum[non_ref]
    obj <- -mean(log_probs)
    obj_values <<- c(obj_values, obj)
    obj
  }
  
  gradient <- function(beta_flat) {
    beta_mat <- matrix(beta_flat, nrow = K-1, ncol = p)
    scores <- X %*% t(beta_mat)
    m <- pmax(0, apply(scores, 1, max))
    scores_centered <- scores - m
    exp_scores <- exp(scores_centered)
    sum_exp <- rowSums(exp_scores) + exp(-m)
    P <- exp_scores / sum_exp
    I <- matrix(0, n, K-1)
    for (k in 1:(K-1)) I[y == k, k] <- 1
    grad_mat <- - (1/n) * t(I - P) %*% X
    as.vector(grad_mat)
  }
  
  result <- optim(beta_init, fn = neg_loglik, gr = gradient,
                  method = "BFGS", control = list(maxit = max_iter, reltol = tol))
  
  obj_values <- obj_values[seq_len(result$counts[1])]
  beta <- matrix(result$par, nrow = K-1, ncol = p)
  
  list(beta = beta, objective = obj_values)
}

# Gradient check
gradient_check <- function(beta_mat, X, y) {
  beta_flat <- as.vector(beta_mat)
  grad_at_final <- gradient(beta_flat, X, y)
  max_abs_grad <- max(abs(grad_at_final))
  list(max_abs_grad = max_abs_grad, tol_satisfied = max_abs_grad < 1e-8)
}
