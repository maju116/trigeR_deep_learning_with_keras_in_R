library(KODAMA)
library(tidyverse)
# Generujemy spiralne dane
dane <- spirals(n = c(200, 200), sd = c(0.4, 0.4)) %>%
  as.tibble() %>%
  bind_cols(class = c(rep(0, 200), rep(1, 200)))
p <- ggplot(dane, aes(x, y, color = as.factor(class))) + geom_point() + theme_bw()
p

# Predykcje z regresji logistycznej
space <- expand.grid(seq(-7, 7, by = 0.1), seq(-7, 7, by = 0.1)) %>%
  rename(x = Var1, y = Var2)
logreg <- glm(class ~ x + y, family = binomial, data = dane)
logreg_preds <- predict(logreg, newdata = space, type = "response")
logreg_preds <- space %>% bind_cols(., class  = factor(ifelse(logreg_preds < 0.5, 0, 1)))
p + geom_point(data = logreg_preds, alpha = 0.1)

# MLP - jedna warstwa ukryta z 2 neuronami
sigmoid <- function(x) {1 / (1 + exp(-x))}

# Propagacja przednia - forward propagation
propagacja_przednia <- function(x, w1, w2) {
  # Kombinacja liniowa inputów i wag
  z1 <- cbind(1, x) %*% w1
  # Funkcja aktywacji w pierwszej warstwie ukrytej
  h <- sigmoid(z1)
  # Kombinacja liniowa neuronów pierwszej warstwy i wag
  z2 <- cbind(1, h) %*% w2
  # Output
  list(output = sigmoid(z2), h = h)
}

# Propagacja wsteczna - forward propagation
propagacja_wsteczna <- function(x, y, y_hat, w1, w2, h, learn_rate) {
  # Gradient funkcji straty względem wag w ostatniej warstwie
  dw2 <- t(cbind(1, h)) %*% (y_hat - y)
  # Gradient względem neuronów w warstwie ukrytej
  dh  <- (y_hat - y) %*% t(w2[-1, , drop = FALSE])
  # Gradient względem wag w pierszej warstwie ukrytej
  dw1 <- t(cbind(1, x)) %*% (h * (1 - h) * dh)
  # Update wag przy pomody Gradient Descent
  w1 <- w1 - learn_rate * dw1
  w2 <- w2 - learn_rate * dw2
  # Nowe wagi
  list(w1 = w1, w2 = w2)
}

# Cała sieć
train <- function(x, y, hidden = 5, learn_rate = 1e-2, iterations = 1e4) {
  # Dodajemy wyraz wolny
  d <- ncol(x) + 1
  # Losowa inicjalizacja wag
  w1 <- matrix(rnorm(d * hidden), d, hidden)
  w2 <- as.matrix(rnorm(hidden + 1))
  # Propaagcja przednia i wsteczna w pętli
  for (i in 1:iterations) {
    ff <- propagacja_przednia(x, w1, w2)
    bp <- propagacja_wsteczna(x, y,
                              y_hat = ff$output,
                              w1, w2,
                              h = ff$h,
                              learn_rate = learn_rate)
    w1 <- bp$w1; w2 <- bp$w2
  }
  # Output i wagi
  list(output = ff$output, w1 = w1, w2 = w2)
}

x <- data.matrix(dane[, c('x', 'y')])
y <- dane$class
mlp5 <- train(x, y, hidden = 5, iterations = 1e5)

mlp_preds <- propagacja_przednia(x = data.matrix(space[, c('x', 'y')]),
                                 w1 = mlp5$w1,
                                 w2 = mlp5$w2)$output
mlp_preds <- space %>% bind_cols(., class  = factor(ifelse(mlp_preds < 0.5, 0, 1)))
p + geom_point(data = mlp_preds, alpha = 0.1)
