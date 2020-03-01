library(tidyverse)
library(keras)
library(scales)

# Boston housing (https://www.kaggle.com/mlg-ulb/creditcardfraud)
boston <- dataset_boston_housing(path = "boston_housing.npz", test_split = 0.2, seed = 113L)
train_X_mean <- boston$train$x %>% apply(., 2, mean)
train_X_sd <- boston$train$x %>% apply(., 2, sd)
boston_train_X <- boston$train$x %>% sweep(., 2, train_X_mean, "-") %>% sweep(., 2, train_X_sd, "/")
boston_train_Y <- boston$train$y
boston_test_X <- boston$test$x %>% sweep(., 2, train_X_mean, "-") %>% sweep(., 2, train_X_sd, "/")
boston_test_Y <- boston$test$y
save(file = "data/boston.RData",
     list = c("boston_train_X", "boston_train_Y",
              "boston_test_X", "boston_test_Y"))

# Binary classification data
bin_class_data <- tibble(
  r = c(runif(800, -0.35, 0.35), runif(400, -1.5, -0.5), runif(400, 0.5, 1.5)),
  t = t <- 2* pi* runif(1600),
  x = r*cos(t),
  y = r*sin(t),
  class = c(rep(0, 800), rep(1, 800))
)
square_data <- expand.grid(seq(-1.5, 1.5, by = 0.01), seq(-1.5, 1.5, by = 0.01)) %>% as.matrix()
save(file = "data/bin_class.RData",
     list = c("bin_class_data", "square_data"))

# Fashion MNIST (https://www.kaggle.com/zalando-research/fashionmnist)
fashion_mnist_train <- read_csv("data/fashion-mnist_train.csv")
fashion_mnist_test <- read_csv("data/fashion-mnist_test.csv")
fashion_mnist_train_X <- fashion_mnist_train %>% select(-label) %>% as.matrix()
fashion_mnist_train_Y <- fashion_mnist_train %>% pull(label)
fashion_mnist_test_X <- fashion_mnist_test %>% select(-label) %>% as.matrix()
fashion_mnist_test_Y <- fashion_mnist_test %>% pull(label)
save(file = "data/fashion_mnist.RData",
     list = c("fashion_mnist_train_X", "fashion_mnist_train_Y",
              "fashion_mnist_test_X", "fashion_mnist_test_Y"))

# Sign MNIST (https://www.kaggle.com/datamunge/sign-language-mnist)
sign_mnist_train <- read_csv("data/sign-language-mnist/sign_mnist_train.csv")
sign_mnist_test <- read_csv("data/sign-language-mnist/sign_mnist_test.csv")
train_path <- "data/sign-language-mnist/train/"
test_path <- "data/sign-language-mnist/test/"
for (lab in unique(sign_mnist_train$label)) {
  dir.create(paste0(train_path, lab))
  dir.create(paste0(test_path, lab))
}
for (ind in 1:nrow(sign_mnist_train)) {
  lab <- sign_mnist_train[ind, 1] %>% as.numeric()
  x <- matrix(as.numeric(sign_mnist_train[ind, -1]), 28, 28, byrow = TRUE)
  png::writePNG(x, paste0(train_path, lab, "/", ind, ".png"))
}
for (ind in 1:nrow(sign_mnist_test)) {
  lab <- sign_mnist_test[ind, 1] %>% as.numeric()
  x <- matrix(as.numeric(sign_mnist_test[ind, -1]), 28, 28, byrow = TRUE)
  png::writePNG(x, paste0(test_path, lab, "/", ind, ".png"))
}

# Creditcard fraud detection (https://www.kaggle.com/mlg-ulb/creditcardfraud)
creditcard <- read_csv("data/creditcard.csv")
index <- sample(1:nrow(creditcard), (nrow(creditcard) * 0.8))
creditcard_train_X <- creditcard %>% select(-Time, -Class) %>% .[index, ] %>% as.matrix()
creditcard_train_X[, 29] <- rescale(creditcard_train_X[, 29], to = c(-1, 1))
# train_X_max <- creditcard_train_X %>% apply(2, max)
# train_X_min <- creditcard_train_X %>% apply(2, min)
# creditcard_train_X <- creditcard_train_X %>% sweep(., 2, train_X_min, "-") %>% sweep(., 2, train_X_max - train_X_min, "/")
creditcard_train_Y <- creditcard %>% .[index, ] %>% pull(Class)
creditcard_test_X <- creditcard %>% select(-Time, -Class) %>% .[-index, ] %>% as.matrix()
creditcard_test_X[, 29] <- rescale(creditcard_test_X[, 29], to = c(-1, 1))
# creditcard_test_X <- creditcard_test_X %>% sweep(., 2, train_X_min, "-") %>% sweep(., 2, train_X_max - train_X_min, "/")
creditcard_test_Y <- creditcard %>% .[-index, ] %>% pull(Class)
save(file = "data/creditcard.RData",
     list = c("creditcard_train_X", "creditcard_train_Y",
              "creditcard_test_X", "creditcard_test_Y"))

# Sentiment140 (https://www.kaggle.com/kazanova/sentiment140)
sentiment140 <- read_csv("data/sentiment140.csv", col_names = FALSE) %>%
  mutate(target = case_when(
    X1 == 0 ~ 0,
    X1 == 4 ~ 1,
    TRUE ~ -99
  )) %>%
  filter(row_number() %% 10 == 0)
sentiment140_Y <- sentiment140$target
sentiment140_X <- sentiment140$X6
save(file = "data/sentiment140.RData", list = c("sentiment140_X", "sentiment140_Y"))
