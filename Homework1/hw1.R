# Logistic Regression with Gradient Descent
LogisticReg.gd <- function(X, y, alpha, max_iter = 1e4, tol = 1e-6) {
  
  # Initialize
  p <- ncol(X)
  beta <- rep(0, p) 
  nll_values <- numeric(max_iter) 
  
  for (iter in 1:max_iter) {
    
    # Probabilities using logistic link
    eta <- X %*% beta
    probs <- exp(eta) / (1 + exp(eta))
    
    # Gradient of negative log-likelihood
    grad <- t(X) %*% (probs - y)
    
    # Update step
    beta <- beta - alpha * grad
    
    # Negative log-likelihood
    nll <- -sum(y * log(probs) + (1 - y) * log(1 - probs))
    nll_values[iter] <- nll
    
    # Convergence check
    if (max(abs(grad)) < tol) {
      nll_values <- nll_values[1:iter]
      break
    }
  }
  
  return(list(beta = beta, nll_values = nll_values))
}

# -----------------------------------------------------
# Simulated Data
set.seed(8054)
n <- 100
p <- 20
X <- cbind(1, matrix(rnorm(n * p), n, p)) # design matrix
beta_true <- c(1, runif(p))
probs <- exp(X %*% beta_true) / (1 + exp(X %*% beta_true))
y <- rbinom(n, 1, probs)

# Step size selection
max_eigen <- max(eigen(t(X) %*% X)$values)
a_values <- 8:1
results <- list()

for (a in a_values) {
  alpha <- (n * a) / max_eigen
  result <- LogisticReg.gd(X, y, alpha)
  results[[as.character(a)]] <- result$nll_values
}

# Collect results for plotting
plot_data <- do.call(rbind, lapply(names(results), function(a) {
  data.frame(iteration = seq_along(results[[a]]), 
             nll = results[[a]], 
             a = as.numeric(a))
}))

# Plot with ggplot2
library(ggplot2)

gg <- ggplot(plot_data, aes(x = iteration, y = nll, color = as.factor(a))) +
  geom_line() +
  labs(title = "Convergence of Gradient Descent for Different Step Sizes",
       x = "Iteration", 
       y = "Negative Log-Likelihood",
       color = "a") +
  theme_minimal()

print(gg)

# Save the plot
ggsave("hw1_results.png", gg, width = 7, height = 5)
