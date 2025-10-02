# Homework 3: Advanced Logistic Regression

## Questions

1. **Constrained Logistic Regression** – `Q1_ConstrainedLogistic.R`  
   Implements an **accelerated projected gradient descent** algorithm to solve  
   $$ \min_\beta \frac{1}{n} \sum_{i=1}^n \left[ \log(1 + e^{x_i^\top \beta}) - y_i (x_i^\top \beta) \right], \quad \text{s.t. } A \beta = b $$  
   Each script contains:
   - Function `LinCon_LogisticReg`
   - Example simulation
   - Plot of objective function convergence (`figures/Q1_obj_convergence.png`)

2. **Multinomial Logistic Regression** – `Q2_MultinomLogistic.R`  
   Implements three optimization algorithms for multinomial logistic regression:
   - **Quasi-Newton (BFGS)**
   - **Mini-batch SGD (MBSGD)**
   - **Stochastic Average Gradient (SAG)**  
   Each script contains:
   - Functions: `QNewton_MultinomLogisticReg`, `MBSGD_MultinomLogisticReg`, `SAG_MultinomLogisticReg`
   - Example simulations
   - Plots:
     - Objective vs. iteration (`figures/Q2_obj_iter.png`)
     - Objective vs. cumulative time (`figures/Q2_obj_time.png`)

## Usage

```r
# Q1
source("Homework3/Q1_ConstrainedLogistic.R")
out1 <- LinCon_LogisticReg(X, y, A, b, alpha, max_iter, tol)

# Q2
source("Homework3/Q2_MultinomLogistic.R")
out2 <- QNewton_MultinomLogisticReg(X, y, max_iter, tol)
out3 <- MBSGD_MultinomLogisticReg(X, y, batch_size, max_iter, tol, gamma, step_size)
out4 <- SAG_MultinomLogisticReg(X, y, max_iter, tol, step_size)
