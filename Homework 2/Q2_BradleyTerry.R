# Q2: Bradley-Terry Model for NBA Rankings
bradley_terry <- function(home.wins, home.losses, max.iter = 10000, tol = 1e-6) {
  stopifnot(nrow(home.wins) == ncol(home.wins),
            all(dim(home.wins) == dim(home.losses)))

  n_teams <- nrow(home.wins)
  team_names <- rownames(home.wins)
  h <- sum(home.wins)
  w <- rowSums(home.wins) + colSums(home.losses)

  theta <- 1
  gamma <- setNames(rep(1, n_teams), team_names)

  iter <- 0
  converged <- FALSE

  while (iter < max.iter && !converged) {
    theta_prev <- theta
    gamma_prev <- gamma

    D <- outer(gamma, gamma, function(gi, gj) theta * gi + gj)
    C <- rowSums((home.wins + home.losses) / D)
    E <- colSums((home.wins + home.losses) / D)

    theta <- h / sum((home.wins + home.losses) * gamma / D)
    gamma <- w / (theta * C + E)

    converged <- max(abs(theta - theta_prev), max(abs(gamma - gamma_prev))) < tol
    iter <- iter + 1
  }

  list(theta = theta, gamma = gamma)
}

# Example usage (NBA data)
library(readxl)
nba_data <- read_excel("nba_data.xlsx", sheet = 1)
team_names <- nba_data$...1  
home.wins <- as.matrix(nba_data[, -1])
rownames(home.wins) <- team_names
colnames(home.wins) <- team_names
home.losses <- t(home.wins)

results <- bradley_terry(home.wins, home.losses)
cat("Estimated Theta (Home-Court Advantage):", results$theta, "\n")
cat("Team Strengths (Gamma) Ranked:\n")
print(sort(results$gamma, decreasing = TRUE))
