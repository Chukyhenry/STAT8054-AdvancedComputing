SAG_MultinomLogisticReg <- function(X, y, max_iter = 1000, tol = 1e-4, step_size = 0.01) {
  X <- as.matrix(X)
  y <- as.integer(y)
  n <- nrow(X); p <- ncol(X); K <- max(y)
  
  beta_flat <- rep(0, (K-1)*p)
  stored_grad <- matrix(0, nrow = n, ncol = (K-1)*p)
  sum_grad <- rep(0, (K-1)*p)
  obj_values <- numeric(max_iter)
  times <- numeric(max_iter)
  
  permuted_indices <- sample(n)
  initial_pass_idx <- 1
  converged <- FALSE
  iter <- 1
  
  while (iter <= max_iter && !converged) {
    start_time <- Sys.time()
    
    if (initial_pass_idx <= n) {
      i <- permuted_indices[initial_pass_idx]
      initial_pass_idx <- initial_pass_idx + 1
      
      beta_mat <- matrix(beta_flat, nrow = K-1, ncol = p)
      x_i <- X[i, ]
      y_i <- y[i]
      scores <- as.vector(x_i %*% t(beta_mat))
      max_score <- max(0, max(scores))
      scores_centered <- scores - max_score
      exp_scores <- exp(scores_centered)
      sum_exp <- sum(exp_scores) + exp(-max_score)
      probs_k <- exp_scores / sum_exp
      I <- numeric(K-1)
      if (y_i < K) I[y_i] <- 1
      grad <- as.vector(tcrossprod(I - probs_k, x_i))
      
      beta_flat <- beta_flat - step_size * grad
      stored_grad[i, ] <- grad
      sum_grad <- sum_grad + grad
    } else {
      i <- sample(n, 1)
      beta_mat <- matrix(beta_flat, nrow = K-1, ncol = p)
      x_i <- X[i, ]; y_i <- y[i]
      scores <- as.vector(x_i %*% t(beta_mat))
      max_score <- max(0, max(scores))
      scores_centered <- scores - max_score
      exp_scores <- exp(scores_centered)
      sum_exp <- sum(exp_scores) + exp(-max_score)
      probs_k <- exp_scores / sum_exp
      I <- numeric(K-1)
      if (y_i < K) I[y_i] <- 1
      new_grad <- as.vector(tcrossprod(I - probs_k, x_i))
      delta <- new_grad - stored_grad[i, ]
      sum_grad <- sum_grad + delta
      stored_grad[i, ] <- new_grad
      avg_grad <- sum_grad / n
      beta_flat <- beta_flat - step_size * avg_grad
    }
    
    beta_mat <- matrix(beta_flat, nrow = K-1, ncol = p)
    scores_full <- X %*% t(beta_mat)
    max_scores <- pmax(0, apply(scores_full, 1, max))
    scores_centered <- scores_full - max_scores
    sum_exp_full <- rowSums(exp(scores_centered)) + exp(-max_scores)
    log_sum <- max_scores + log(sum_exp_full)
    log_probs <- numeric(n)
    is_ref <- (y == K)
    log_probs[is_ref] <- -log_sum[is_ref]
    non_ref <- which(!is_ref)
    y_non_ref <- y[non_ref]
    log_probs[non_ref] <- scores_centered[cbind(non_ref, y_non_ref)] - log_sum[non_ref]
    obj_values[iter] <- -mean(log_probs)
    
    if (initial_pass_idx > n) {
      avg_grad <- sum_grad / n
      grad_norm <- norm(matrix(avg_grad, nrow = K-1), "F")
      if (grad_norm < tol) converged <- TRUE
    }
    
    times[iter] <- as.numeric(Sys.time() - start_time)
    iter <- iter + 1
  }
  
  iter_used <- min(iter - 1, max_iter)
  list(beta = matrix(beta_flat, nrow = K-1, ncol = p),
       obj = obj_values[1:iter_used],
       time = times[1:iter_used])
}
