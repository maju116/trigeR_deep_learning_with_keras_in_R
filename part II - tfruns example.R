# Create 'flags' - hyperparameters we want to check for different values
FLAGS <- flags(
  flag_integer("units", default = 4),
  flag_string("activation", default = "relu"),
  flag_numeric("dropout", default = 0.3),
  flag_integer("batch_size", default = 50)
)

# Define model with flags
bin_model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$units, input_shape = c(2), activation = FLAGS$activation) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
layer_dense(units = 1, activation = "sigmoid")

bin_model %>% compile(loss = 'binary_crossentropy',
                      optimizer = 'sgd',
                      metrics = c('accuracy'))

history <- bin_model %>% fit(x = bin_class_train_X,
                             y = bin_class_train_Y,
                             validation_data = list(bin_class_test_X, bin_class_test_Y),
                             epochs = 100,
                             batch_size = FLAGS$batch_size,
                             view_metrics = FALSE) # Don't enerate plot in viewer
