## part I - Keras, Tensorflow and MLP
# bin_model
bin_model <- keras_model_sequential() %>%
  layer_dense(8, input_shape = c(2), activation = "relu",
              activity_regularizer = regularizer_l1(l = 0.3)) %>%
  layer_dense(1, activation = "sigmoid") %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'sgd',
          metrics = c('accuracy'))
history <- bin_model %>% fit(x = bin_class_train_X,
                             y = bin_class_train_Y,
                             validation_split = 0.1,
                             epochs = 100,
                             batch_size = 100,
                             verbose = 0)

bin_model %>% evaluate(bin_class_test_X, bin_class_test_Y)

# fashion_mnist_model
fashion_mnist_train_Y <- fashion_mnist_train_Y %>% to_categorical(., 10)
fashion_mnist_test_Y <- fashion_mnist_test_Y  %>% to_categorical(., 10)

fashion_mnist_train_X <- fashion_mnist_train_X / 255
fashion_mnist_test_X <- fashion_mnist_test_X/ 255

fashion_model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 10, activation = "softmax")

fashion_model %>% compile(
  optimizer = "sgd",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- fashion_model %>%
  fit(x = fashion_mnist_train_X,
      y = fashion_mnist_train_Y,
      validation_split = 0.2,
      epochs = 20,
      batch_size = 128)

fashion_model %>% evaluate(fashion_mnist_test_X, fashion_mnist_test_Y)

# model_checkpoint
bin_model <- keras_model_sequential() %>%
  layer_dense(8, input_shape = c(2), activation = "relu",
              activity_regularizer = regularizer_l1(l = 0.3)) %>%
  layer_dense(1, activation = "sigmoid") %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'sgd',
          metrics = c('accuracy'))
model_checkpoint <- callback_model_checkpoint(filepath = "models/bin_model.{epoch:02d}-{val_loss:.2f}.hdf5",
                                              monitor = "val_loss",
                                              period = 1)
history <- bin_model %>% fit(x = bin_class_train_X,
                             y = bin_class_train_Y,
                             validation_split = 0.1,
                             epochs = 100,
                             batch_size = 100,
                             verbose = 1,
                             callbacks = model_checkpoint)

# early_stopping
bin_model <- keras_model_sequential() %>%
  layer_dense(4, input_shape = c(2), activation = "tanh",
              activity_regularizer = regularizer_l1(l = 0.3)) %>%
  layer_dense(1, activation = "sigmoid") %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'sgd',
          metrics = c('accuracy'))
model_checkpoint <- callback_model_checkpoint(filepath = "models/bin_model.best.hdf5",
                                              monitor = "val_acc",
                                              period = 1)
early_stopping <- callback_early_stopping(monitor = "val_acc", patience = 3)
history <- bin_model %>% fit(x = bin_class_train_X,
                             y = bin_class_train_Y,
                             validation_split = 0.1,
                             epochs = 100,
                             batch_size = 100,
                             verbose = 1,
                             callbacks = list(model_checkpoint, early_stopping))

## part III - Stochastic Gradient Descent and Backpropagation
# ordinary_function
f <- function(x) x^2 + 1
grad_f <- function(x) 2*x

# ordinary_function_GD
x <- x - lr * grad_f(x)

# linear_reg_r
lm_model <- lm(y ~ x, sample_data)

# linear_reg_matrix_data
X <- tibble(x0 = 1, x1 = sample_data$x) %>% as.matrix()
y <- sample_data$y

# linear_reg_mse
MSE <- function(beta, X, y) mean((beta%*%t(X) - y)^2)

# linear_reg_mse_grad
MSE_grad <- function(beta, X, y) 2*((beta%*%t(X) - y)%*%X)/length(y)

# linear_reg_gradient_descent
beta <- beta - lr * MSE_grad(beta, X, y)

# linear_reg_stochastic_gradient_descent
batches_per_epoch <- ceiling(length(y) / batch_size)
indexes <- ((b - 1) * batch_size + 1):min((b * batch_size), length(y))
X_b <- X[indexes, , drop = FALSE]
y_b <- y[indexes]
beta <- beta - lr * MSE_grad(beta, X_b, y_b)

# logistic_reg_r
logistic_model <- glm(class ~ x + y, sample_data, family = "binomial")

# logistic_reg_matrix_data
X <- tibble(x0 = 1, x1 = sample_data$x, x2 = sample_data$y) %>% as.matrix()
y <- sample_data$class

# sigmoid
sigmoid <- function(x) 1 / (1 + exp(-x))

# sigmoid_grad
sigmoid_grad <- function(x) sigmoid(x) * (1 - sigmoid(x))

# binary_crossentropy
binary_crossentropy <- function(beta, X, y) {
  z <- sigmoid(beta%*%t(X))
  -mean(y * log(z) + (1 - y) * log(1 - z))
}

# binary_crossentropy_grad
binary_crossentropy_grad <- function(beta, X, y) {
  z <- sigmoid(beta%*%t(X))
  dL <- (-y / z - (1 - y) / (z - 1)) / length(y)
  dV <- sigmoid_grad(beta%*%t(X))
  dx <- X
  (dL * dV) %*% dx
}

# logistic_reg_sgd
batches_per_epoch <- ceiling(length(y) / batch_size)
indexes <- ((b - 1) * batch_size + 1):min((b * batch_size), length(y))
X_b <- X[indexes, , drop = FALSE]
y_b <- y[indexes]
beta <- beta - lr * binary_crossentropy_grad(beta, X_b, y_b)


# forward_step
forward_propagation <- function(X, w1, w2) {
  # Linear combination of inputs and weights
  z1 <- X %*% w1
  # Activation function - sigmoid
  h <- sigmoid(z1)
  # Linear combination of 1-layer hidden units and weights
  z2 <- cbind(1, h) %*% w2
  # Output
  list(output = sigmoid(z2), h = h)
}

# backward_step
backward_propagation <- function(X, y, y_hat, w1, w2, h, lr) {
  # w2 gradient
  dw2 <- t(cbind(1, h)) %*% (y_hat - y)
  # h gradient
  dh  <- (y_hat - y) %*% t(w2[-1, , drop = FALSE])
  # w1 gradient
  dw1 <- t(X) %*% ((h * (1 - h) * dh))
  # SGD
  w1 <- w1 - lr * dw1
  w2 <- w2 - lr * dw2
  list(w1 = w1, w2 = w2)
}

# single_layer_perceptron_sgd
ff <- forward_propagation(X_b, w1, w2)
bp <- backward_propagation(X_b, y_b,
                           y_hat = ff$output,
                           w1, w2,
                           h = ff$h,
                           lr = lr)

## part IV - Convolutional Neural Networks
# fashion_mnist_ex
fmnist_model2 <- keras_model_sequential() %>%
  layer_conv_2d(
    filter = 64, kernel_size = c(3, 3), padding = "same",
    input_shape = c(28, 28, 1), activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(10, activation = "softmax")

fmnist_model2 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adamax(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

if (!dir.exists("tensorboard")) dir.create("tensorboard")
history <- fmnist_model2 %>% fit(
  fashion_mnist_train_X,
  fashion_mnist_train_Y,
  batch_size = 128,
  epochs = 10,
  validation_split = 0.2,
  callbacks = c(callback_model_checkpoint(monitor = "val_acc",
                                          filepath = "models/fmnist_model1.hdf5",
                                          save_best_only = TRUE),
                callback_early_stopping(monitor = "val_loss", patience = 5),
                callback_tensorboard(log_dir = "tensorboard"))
)

tensorboard("tensorboard")
