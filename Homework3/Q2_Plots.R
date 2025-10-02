library(ggplot2)

plot_results <- function(results, batch_sizes, save_figures = TRUE) {
  plot_data <- data.frame()
  
  for (bs in batch_sizes) {
    out <- results[[paste0("MBSGD_", bs)]]
    df <- data.frame(
      Iteration = 1:length(out$obj),
      Objective = out$obj,
      Time = cumsum(out$time),
      Algorithm = paste0("MBSGD (batch=", bs, ")")
    )
    plot_data <- rbind(plot_data, df)
  }
  
  # Add SAG
  out_sag <- results[["SAG"]]
  df_sag <- data.frame(
    Iteration = 1:length(out_sag$obj),
    Objective = out_sag$obj,
    Time = cumsum(out_sag$time),
    Algorithm = "SAG"
  )
  plot_data <- rbind(plot_data, df_sag)
  
  p1 <- ggplot(plot_data, aes(x = Iteration, y = Objective, color = Algorithm)) +
    geom_line() +
    labs(title = "Objective Function Convergence", x = "Iteration", y = "Negative Log-Likelihood")
  
  p2 <- ggplot(plot_data, aes(x = Time, y = Objective, color = Algorithm)) +
    geom_line() +
    labs(title = "Objective vs. Elapsed Time", x = "Time (seconds)", y = "Negative Log-Likelihood")
  
  if (save_figures) {
    ggsave("figures/Q2_obj_iter.png", plot = p1, width = 8, height = 6)
    ggsave("figures/Q2_obj_time.png", plot = p2, width = 8, height = 6)
  }
  
  list(plot_iteration = p1, plot_time = p2)
}
