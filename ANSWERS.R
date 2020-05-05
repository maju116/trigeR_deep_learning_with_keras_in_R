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
  loss = "categorical_crossentropy",
  optimizer = optimizer_adadelta(lr = 0.01, decay = 1e-6),
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
                                          filepath = "models/fmnist_model2.hdf5",
                                          save_best_only = TRUE),
                callback_early_stopping(monitor = "val_acc", patience = 5),
                callback_tensorboard(log_dir = "tensorboard"))
)

tensorboard("tensorboard")

fmnist_model2 %>% evaluate(fashion_mnist_test_X, fashion_mnist_test_Y)

## part V - Fine-tuning and data generators
# alien_predator_model_1
alien_predator_model_1 <- keras_model_sequential() %>%
  layer_conv_2d(
    filter = 64, kernel_size = c(3, 3), padding = "same",
    input_shape = c(150, 150, 3), activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(2, activation = "softmax")

alien_predator_model_1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy"))

# sign_mnist_ex
train_path <- "data/sign-language-mnist/train/"
test_path <- "data/sign-language-mnist/test/"
train_datagen <- image_data_generator(
  rescale = 1/255, # changes pixel range from [0, 255] to [0, 1]
  rotation_range = 10,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  zoom_range = 0.1,
  horizontal_flip = FALSE
)

validation_datagen <- image_data_generator(rescale = 1/255)

train_flow <- flow_images_from_directory(
  directory = train_path,
  generator = train_datagen,
  color_mode = "grayscale",
  target_size = c(28, 28),
  batch_size = 32,
  class_mode = "categorical"
)

validation_flow <- flow_images_from_directory(
  directory = test_path,
  generator = validation_datagen,
  color_mode = "grayscale",
  target_size = c(28, 28),
  batch_size = 32,
  class_mode = "categorical"
)

sign_mnist_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),  activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_dropout(rate = 0.40) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 24, activation = 'softmax')

sign_mnist_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = "accuracy"
)

history <- sign_mnist_model %>% fit_generator(
  train_flow,
  steps_per_epoch = 858,
  epochs = 15,
  validation_data = validation_flow,
  validation_steps = 225
)

## part VI - Autoencoders
# denoising_autoencoder_mnist
  # Eccoder
autoencoder <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                input_shape = c(28, 28, 1), padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2, 2), padding = 'same') %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2, 2), padding = 'same') %>%
   # Decoder
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                padding = 'same') %>%
  layer_upsampling_2d(size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                padding = 'same') %>%
  layer_upsampling_2d(size = c(2, 2)) %>%
  layer_conv_2d(filters = 1, kernel_size = c(3, 3), activation = 'sigmoid',
                padding = 'same')

# denoising_autoencoder_mnist_loss
autoencoder %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam'
)

# denoising_autoencoder_mnist_fit
history <- autoencoder %>%
  fit(x = mnist_train_X_noise,
      y = mnist_train_X,
      epochs = 30,
      batch_size = 128,
      validation_split = 0.2
  )

# denoising_autoencoder_mnist_predict
autoencoder_predictions <- autoencoder %>% predict(mnist_test_X_noise)

# mnist_reshape
mnist_train_X_vec <- array_reshape(mnist_train_X, c(60000, 784))
mnist_test_X_vec <- array_reshape(mnist_test_X, c(10000, 784))

# mnist_dr_autoencoder
input <- layer_input(shape = c(784))

encoder <- input %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(32, activation = 'relu')

decoder <- encoder %>%
  layer_dense(64, activation = 'relu') %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dense(784, activation = 'relu')

autoencoder <- keras_model(input, decoder)
encoder_model <- keras_model(input, encoder)

# mnist_dr_autoencoder_compile
autoencoder %>% compile(
  loss = 'mse',
  optimizer = 'adam'
)

# mnist_dr_autoencoder_fit
history <- autoencoder %>%
  fit(x = mnist_train_X_vec,
      y = mnist_train_X_vec,
      epochs = 30,
      batch_size = 128,
      validation_split = 0.2
  )

# creditcard_am_autoencoder
autoencoder <- keras_model_sequential() %>%
  layer_dense(14, activation = 'tanh', input_shape = c(29),
              activity_regularizer = regularizer_l1(10e-5)) %>%
  layer_dense(7, activation = 'relu') %>%
  layer_dense(7, activation = 'tanh') %>%
  layer_dense(29, activation = 'relu')

# creditcard_am_autoencoder_compile
autoencoder %>% compile(
  loss = 'mse',
  optimizer = 'adam'
)

# creditcard_am_autoencoder_fit
history <- autoencoder %>%
  fit(x = creditcard_train_X,
      y = creditcard_train_X,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2
  )

## part VII - Recurrent Neural Networks
# sentiment140_lstm
model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = 20000,
                  output_dim = 128, # Represent each word in 128-dim space
                  input_length = maxlen) %>%
  layer_lstm(units = 15, recurrent_dropout = 0.5, return_sequences = TRUE) %>%
  layer_lstm(units = 7, recurrent_dropout = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model2 %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history2 <- model2 %>% fit(
  sentiment140_X_train,
  sentiment140_Y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

# so_questions
stack_overflow_Y <- stack_overflow_Y %>% to_categorical(20)

tokenizer <- text_tokenizer(
  num_words = 100000, # Max number of unique words to keep
  filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", # Signs to filter out from text
  lower = TRUE, # Schould everything be converted to lowercse
  split = " ", # Token splitting character
  char_level = FALSE, # Should each sign be a token
  oov_token = NULL # Token to replace out-of-vocabulary words
)
tokenizer %>% fit_text_tokenizer(stack_overflow_X)

sequences <- texts_to_sequences(tokenizer, stack_overflow_X)

maxlen <- 30
sequences_pad <- pad_sequences(sequences, maxlen = maxlen)

train_factor <- sample(1:nrow(sequences_pad), nrow(sequences_pad) * 0.8)
stack_overflow_X_train <- sequences_pad[train_factor, ]
stack_overflow_X_test <- sequences_pad[-train_factor, ]
stack_overflow_Y_train <- stack_overflow_Y[train_factor, ]
stack_overflow_Y_test <- stack_overflow_Y[-train_factor, ]

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 100000,
                  output_dim = 300,
                  input_length = maxlen) %>%
  layer_lstm(units = 300, recurrent_dropout = 0.5, return_sequences = TRUE) %>%
  layer_lstm(units = 150, recurrent_dropout = 0.5, return_sequences = TRUE) %>%
  layer_lstm(units = 75, recurrent_dropout = 0.5) %>%
  layer_dense(units = 20, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  stack_overflow_X_train,
  stack_overflow_Y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)
