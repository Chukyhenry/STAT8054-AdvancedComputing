library(MASS)

# Ensure figures folder exists
if(!dir.exists("figures")) dir.create("figures")

LinCon_LogisticReg <- function(X, y, A, b, alpha, max_iter, tol) {
  project <- function(z) {
    AA <- A %*% t(A)  
    residual <- A %*% z - b  
    lambda <- tryCatch({
      solve(AA, residual)
    }, error = function(e) {
      MASS::ginv(AA) %*% residual  
    })
    return(z - t(A) %*% lambda) 
  }
  
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)  
  beta_prev <- rep(0, p)
  t <- 1
  obj_values <- numeric(max_iter)
  
  Xt <- t(X)
  
  logistic_loss <- function(beta) {
    linear_combination <- X %*% beta
    mean(log(1 + exp(linear_combination)) - y * linear_combination)
  }
  
  gradient <- function(beta) {
    linear_combination <- X %*% beta
    prob <- 1 / (1 + exp(-linear_combination))  
    return((Xt %*% (prob - y)) / n)
  }
  
  for (iteration in 1:max_iter) {
    grad <- gradient(beta)
    beta_new <- beta - alpha * grad
    beta_new <- project(beta_new)
    t_new <- (1 + sqrt(1 + 4 * t^2)) / 2
    beta_accelerated <- beta_new + ((t - 1) / t_new) * (beta_new - beta_prev)
    obj_values[iteration] <- logistic_loss(beta_accelerated)
    
    if (iteration > 1 && abs(obj_values[iteration] - obj_values[iteration - 1]) / abs(obj_values[iteration - 1]) < tol) {
      obj_values <- obj_values[1:iteration] 
      break
    }
    
    beta_prev <- beta
    beta <- beta_accelerated
    t <- t_new
  }
  
  return(list(beta = beta, objective_values = obj_values))
}

# Example simulation
set.seed(8054)
n <- 200; p <- 100; m <- 20
X <- matrix(rnorm(n * p), n, p)
A <- matrix(rnorm(m * p), m, p)
b <- runif(m, 3, 10)
beta0 <- ginv(A) %*% b
beta <- beta0 + runif(p, -1, 1)
y <- rbinom(n, 1, 1/(1 + exp(-X %*% beta)))

max_iter <- 5000
tol <- 1e-06
alpha <- 4/(max(eigen(crossprod(X))$values))

out <- LinCon_LogisticReg(X, y, A, b, alpha, max_iter, tol)

# Save plot
png("figures/Q1_obj_convergence.png", width = 800, height = 600)
plot(out$objective_values, type = "l",
     main = "Objective Function Convergence",
     xlab = "Iteration", ylab = "Objective Function Value")
dev.off()
