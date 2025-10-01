library(MASS)
library(ggplot2)

# Quasi-Newton (BFGS) Multinomial Logistic Regression
QNewton_MultinomLogisticReg <- function(X, y, max_iter, tol) {
  X <- as.matrix(X)
  y <- as.integer(y)
  n <- nrow(X); p <- ncol(X); K <- max(y)
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
    return(obj)
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
    return(as.vector(grad_mat))
  }
  
  result <- optim(par = beta_init, fn = neg_loglik, gr = gradient, method = "BFGS",
                  control = list(maxit = max_iter, reltol = tol))
  obj_values <- obj_values[seq_len(result$counts[1])]
  beta <- matrix(result$par, nrow = K-1, ncol = p)
  return(list(beta = beta, objective = obj_values))
}

# Mini-batch SGD
MBSGD_MultinomLogisticReg <- function(X, y, batch_size, max_iter, tol, gamma, step_size) {
  # [Use the function you already wrote; keep the same interface]
}

# SAG
SAG_MultinomLogisticReg <- function(X, y, max_iter, tol, step_size) {
  # [Use the function you already wrote; keep the same interface]
}

# Example usage
set.seed(8054)
n <- 25000; p <- 100; K <- 6
X <- matrix(rnorm(n * p), n, p)
beta <- matrix(rnorm(p * K), p, K)
xb <- crossprod(t(X), beta)
probs <- exp(xb) / rowSums(exp(xb))
y <- sapply(1:n, function(j) sample(1:K, 1, prob = probs[j, ]))

max_iter <- 1000; tol <- 1e-4; gamma <- 0.1; step_size <- 0.01
batch_sizes <- c(1, 10, 50, 100)
results <- list()

for (bs in batch_sizes) {
  results[[paste0("MBSGD_", bs)]] <- MBSGD_MultinomLogisticReg(X, y, bs, max_iter, tol, gamma, step_size)
}

results[["SAG"]] <- SAG_MultinomLogisticReg(X, y, max_iter, tol, step_size)

# Plot Objective vs Iteration
plot_data <- data.frame()
for (bs in batch_sizes) {
  out <- results[[paste0("MBSGD_", bs)]]
  df <- data.frame(Iteration = 1:length(out$obj), Objective = out$obj,
                   Time = cumsum(out$time), Algorithm = paste0("MBSGD (batch=", bs, ")"))
  plot_data <- rbind(plot_data, df)
}
df_sag <- data.frame(Iteration = 1:length(results[["SAG"]]$obj), Objective = results[["SAG"]]$obj,
                     Time = cumsum(results[["SAG"]]$time), Algorithm = "SAG")
plot_data <- rbind(plot_data, df_sag)

ggplot(plot_data, aes(x = Iteration, y = Objective, color = Algorithm)) + geom_line() +
  labs(title = "Objective Function Convergence", x = "Iteration", y = "Negative Log-Likelihood")
