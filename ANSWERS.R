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
  layer_dense(4, input_shape = c(2), activation = "relu",
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
