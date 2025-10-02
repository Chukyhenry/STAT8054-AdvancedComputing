MBSGD_MultinomLogisticReg <- function(X, y, batch_size, max_iter = 1000, tol = 1e-4, gamma = 0.1, step_size = 0.01) {
  X <- as.matrix(X)
  y <- as.integer(y)
  n <- nrow(X); p <- ncol(X); K <- max(y)
  
  beta_flat <- rep(0, (K-1)*p)
  ema_grad_norm <- 0
  obj_values <- numeric(max_iter)
  times <- numeric(max_iter)
  
  iter <- 1
  converged <- FALSE
  
  while (iter <= max_iter && !converged) {
    start_time <- Sys.time()
    
    batch_ind <- sample(n, batch_size, replace = TRUE)
    X_batch <- X[batch_ind, , drop = FALSE]
    y_batch <- y[batch_ind]
    
    beta_mat <- matrix(beta_flat, nrow = K-1, ncol = p)
    scores <- X_batch %*% t(beta_mat)
    max_scores <- apply(scores, 1, max)
    scores_centered <- scores - max_scores
    exp_scores <- exp(scores_centered)
    sum_exp <- rowSums(exp_scores) + exp(-max_scores)
    P <- exp_scores / sum_exp
    
    I <- matrix(0, batch_size, K-1)
    for (k in 1:(K-1)) I[y_batch == k, k] <- 1
    
    grad_mat <- (1 / batch_size) * t(I - P) %*% X_batch
    grad_flat <- as.vector(grad_mat)
    grad_norm <- norm(grad_mat, "F")
    
    ema_grad_norm <- ifelse(iter == 1, grad_norm, gamma * grad_norm + (1 - gamma) * ema_grad_norm)
    
    beta_flat <- beta_flat - step_size * grad_flat
    
    log_probs <- numeric(batch_size)
    is_ref <- (y_batch == K)
    log_probs[is_ref] <- -log(sum_exp[is_ref])
    non_ref <- which(!is_ref)
    y_non_ref <- y_batch[non_ref]
    log_probs[non_ref] <- scores_centered[cbind(non_ref, y_non_ref)] - log(sum_exp[non_ref])
    obj_values[iter] <- -mean(log_probs)
    
    if (ema_grad_norm < tol) converged <- TRUE
    times[iter] <- as.numeric(Sys.time() - start_time)
    iter <- iter + 1
  }
  
  iter_used <- min(iter - 1, max_iter)
  list(beta = matrix(beta_flat, nrow = K-1, ncol = p),
       obj = obj_values[1:iter_used],
       time = times[1:iter_used])
}
