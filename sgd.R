library(tidyverse)
library(gridExtra)
library(KODAMA)
# Ordinary function y=x^2 GD
f <- function(x) x^2
grad_f <- function(x) 2*x
sample_data = tibble(
  x = seq(-2, 2, by = 0.05),
  y = f(x),
  grad = grad_f(x)
)
base_plot <- ggplot(sample_data, aes(x, y)) + geom_line(color = "red") + geom_line(aes(y = grad), color = "blue") +
  theme_bw()
base_plot

x0 <- 1.345 # Start point
lr <- 0.1 # Learning rate
epochs <- 30 # Nr of epochs
GD_ordinary_fun <- function(f, grad_f, x0, lr, epochs) {
  x <- x0
  results <- tibble(
    x = x0, y = f(x), grad = grad_f(x)
  )
  for (i in 1:epochs) {
    x <- x - lr * grad_f(x) # GD
    results[i + 1, ] <- c(x, f(x), grad_f(x))
    print(paste("Updated x value:", round(x, 8), ". Updated f(x) value:", round(f(x), 8)))
  }
  plot(base_plot + geom_point(data = results, color = "black"))
  results
}
task1 <- GD_ordinary_fun(f, grad_f, x0, lr, epochs)

# Linear regression GD
set.seed(666)
sample_data <- tibble(x = runif(50, -3, 3), y = x + rnorm(50, 3, 1.2))
base_plot <- ggplot(sample_data, aes(x, y)) + geom_point() + theme_bw()
base_plot
lm_model <- lm(y ~ x, sample_data)
summary(lm_model)
mean(lm_model$residuals^2) # MSE from model
X <- tibble(x0 = 1, x1 = sample_data$x) %>% as.matrix()
y <- sample_data$y
solve(t(X)%*%X)%*%t(X)%*%y # Numerical solution
MSE <- function(beta, X, y) mean((beta%*%t(X) - y)^2)
MSE(c(2.95039, 0.93806), X, y)
MSE_grad <- function(beta, X, y) 2*((beta%*%t(X) - y)%*%X)/length(y)

beta00 <- c(5, -0.2) # Start point
lr <- 0.1 # Learning rate
epochs <- 30 # Nr of epochs
GD_linear_regression <- function(beta00, X, y, lr, epochs) {
  beta <- beta00
  results <- tibble(
    b0 = beta00[1], b1 = beta00[2], mse = MSE(beta00, X, y), epoch = 0
  )
  for (i in 1:epochs) {
    beta <- beta - lr * MSE_grad(beta, X, y) # GD
    results[i + 1, ] <- c(beta[1], beta[2], MSE(beta, X, y), i)
    print(paste("Updated b0 value:", round(beta[1], 8),
                "Updated b1 value:", round(beta[2], 8),
                "Updated MSE value:", round(MSE(beta, X, y), 8)))
  }
  # Diagnostic plots
  p1 <- base_plot + geom_abline(intercept = beta00[1], slope = beta00[2], color = "blue") +
    geom_abline(intercept = beta[1], slope = beta[2], color = "red")
  p2 <- ggplot(results, aes(epoch, mse)) + theme_bw() + geom_line(color = "red")
  lin_space <- expand.grid(seq(beta[1] - 2, beta[1] + 2, by = 0.1),
                           seq(beta[2] - 2, beta[2] + 2, by = 0.1)) %>%
    as.data.frame() %>% set_names(c("b0", "b1")) %>% rowwise() %>%
    mutate(mse = MSE(c(b0, b1), X, y))
  p3 <- ggplot(lin_space, aes(b0, b1)) + theme_bw() + geom_raster(aes(fill = mse)) +
    geom_contour(colour = "white", aes(z = mse)) + scale_fill_gradient(low = "blue", high = "red") +
    geom_line(data = results, color = "black", linetype = "dashed") +
    geom_point(data = results, color = "black")
  plot(grid.arrange(p1, p2, p3, layout_matrix = rbind(c(1, 2), c(3, 3))))
  results
}
task2 <- GD_linear_regression(beta00, X, y, lr, epochs)

# Linear regression SGD
set.seed(666)
sample_data <- tibble(x = runif(50, -3, 3), y = x + rnorm(50, 3, 1.2))
base_plot <- ggplot(sample_data, aes(x, y)) + geom_point() + theme_bw()
base_plot
X <- tibble(x0 = 1, x1 = sample_data$x) %>% as.matrix()
y <- sample_data$y
MSE <- function(beta, X, y) mean((beta%*%t(X) - y)^2)
MSE_grad <- function(beta, X, y) 2*((beta%*%t(X) - y)%*%X)/length(y)

beta00 <- c(5, -0.2) # Start point
lr <- 0.1 # Learning rate
epochs <- 50 # Nr of epochs
batch_size <- 3 # Batch size
SGD_linear_regression <- function(beta00, X, y, lr, epochs, batch_size) {
  beta <- beta00
  results <- tibble(
    b0 = beta00[1], b1 = beta00[2], mse = MSE(beta00, X, y), epoch = 0
  )
  batches_per_epoch <- ceiling(length(y) / batch_size)
  for (i in 1:epochs) {
    for (b in 1:batches_per_epoch) {
      indexes <- ((b - 1) * batch_size + 1):min((b * batch_size), length(y))
      X_b <- X[indexes, , drop = FALSE]
      y_b <- y[indexes]
      beta <- beta - lr * MSE_grad(beta, X_b, y_b) # SGD
      results <- rbind(results, c(beta[1], beta[2], MSE(beta, X, y), i + b / batches_per_epoch))
    }
    print(paste("Updated b0 value:", round(beta[1], 8),
                "Updated b1 value:", round(beta[2], 8),
                "Updated MSE value:", round(MSE(beta, X, y), 8)))
  }
  # Diagnostic plots
  p1 <- base_plot + geom_abline(intercept = beta00[1], slope = beta00[2], color = "blue") +
    geom_abline(intercept = beta[1], slope = beta[2], color = "red")
  p2 <- ggplot(results, aes(epoch, mse)) + theme_bw() + geom_line(color = "red")
  lin_space <- expand.grid(seq(beta[1] - 2, beta[1] + 2, by = 0.1),
                           seq(beta[2] - 2, beta[2] + 2, by = 0.1)) %>%
    as.data.frame() %>% set_names(c("b0", "b1")) %>% rowwise() %>%
    mutate(mse = MSE(c(b0, b1), X, y))
  p3 <- ggplot(lin_space, aes(b0, b1)) + theme_bw() + geom_raster(aes(fill = mse)) +
    geom_contour(colour = "white", aes(z = mse)) + scale_fill_gradient(low = "blue", high = "red") +
    geom_line(data = results, color = "black", linetype = "dashed") +
    geom_point(data = results, color = "black")
  plot(grid.arrange(p1, p2, p3, layout_matrix = rbind(c(1, 2), c(3, 3))))
  results
}
task3 <- SGD_linear_regression(beta00, X, y, lr, epochs, batch_size)

# Logistic regression SGD
set.seed(666)
sample_data <- spirals(n = c(200, 200), sd = c(0.4, 0.4)) %>%
  as_tibble() %>%
  bind_cols(class = c(rep(0, 200), rep(1, 200))) %>%
  .[sample(1:400, 400), ]
base_plot <- ggplot(sample_data, aes(x, y, color = as.factor(class))) + geom_point() + theme_bw()
base_plot
logistic_model <- glm(class ~ x + y, sample_data, family = "binomial")
summary(logistic_model)
X <- tibble(x0 = 1, x1 = sample_data$x, x2 = sample_data$y) %>% as.matrix()
y <- sample_data$class
sigmoid <- function(x) 1 / (1 + exp(-x))
sigmoid_grad <- function(x) sigmoid(x) * (1 - sigmoid(x))
binary_crossentropy <- function(beta, X, y) {
  z <- sigmoid(beta%*%t(X))
  -mean(y * log(z) + (1 - y) * log(1 - z))
}
binary_crossentropy_grad <- function(beta, X, y) {
  z <- sigmoid(beta%*%t(X))
  dL <- (-y / z - (1 - y) / (z - 1)) / length(y)
  dV <- sigmoid_grad(beta%*%t(X))
  dx <- X
  (dL * dV) %*% dx
}

beta00 <- c(0.3, 1, -0.2) # Start point
lr <- 0.1 # Learning rate
epochs <- 50 # Nr of epochs
batch_size <- 50 # Batch size
SGD_logistic_regression <- function(beta00, X, y, lr, epochs, batch_size) {
  beta <- beta00
  results <- tibble(
    b0 = beta00[1], b1 = beta00[2], b2 = beta00[3], log_loss = binary_crossentropy(beta00, X, y), epoch = 0
  )
  batches_per_epoch <- ceiling(length(y) / batch_size)
  for (i in 1:epochs) {
    for (b in 1:batches_per_epoch) {
      indexes <- ((b - 1) * batch_size + 1):min((b * batch_size), length(y))
      X_b <- X[indexes, , drop = FALSE]
      y_b <- y[indexes]
      beta <- beta - lr * binary_crossentropy_grad(beta, X_b, y_b) # SGD
      results <- rbind(results, c(beta[1], beta[2], beta[3], binary_crossentropy(beta, X, y), i + b / batches_per_epoch))
    }
    print(paste("Updated b0 value:", round(beta[1], 8),
                "Updated b1 value:", round(beta[2], 8),
                "Updated b2 value:", round(beta[3], 8),
                "Updated LogLoss value:", round(binary_crossentropy(beta, X, y), 8)))
  }
  # Diagnostic plots
  p1 <- ggplot(results, aes(epoch, log_loss)) + theme_bw() + geom_line(color = "red")
  lin_space <- expand.grid(seq(-6, 6, by = 0.1), seq(-6, 6, by = 0.1)) %>%
    as.data.frame() %>% set_names(c("x", "y")) %>% rowwise() %>%
    mutate(proba = sigmoid(beta%*%c(1, x, y)),
           class = ifelse(proba > 0.5, 1, 0))
  p2 <- base_plot + geom_point(data = lin_space, alpha = 0.1)
  plot(grid.arrange(p1, p2, ncol = 2))
  results
}
task4 <- SGD_logistic_regression(beta00, X, y, lr, epochs, batch_size)

