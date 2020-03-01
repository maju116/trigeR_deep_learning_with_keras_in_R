library(keras)
library(tidyverse)

# PRZYKŁAD
# Analiza sentymentu dla zbioru tweetów
load("data/sentiment140.RData")

# Tworzymy tokenizer - zmiana tekstu na tokeny
tokenizer <- text_tokenizer(num_words = 20000)
tokenizer %>% fit_text_tokenizer(sentiment140_X)

# Zmiana tekstu na sekwencje tokenów
sequences <- texts_to_sequences(tokenizer, sentiment140_X)

# Przycinanie sekwencji do długości 20 tokenów
maxlen <- 20
sequences_pad <- pad_sequences(sequences, maxlen = maxlen)

# Podział na zbiór treningowy i testowy
train_factor <- sample(1:nrow(sequences_pad), nrow(sequences_pad) * 0.8)
sentiment140_X_train <- sequences_pad[train_factor, ]
sentiment140_X_test <- sequences_pad[-train_factor, ]
sentiment140_Y_train <- sentiment140_Y[train_factor]
sentiment140_Y_test <- sentiment140_Y[-train_factor]

# Budowa modelu wraz z embedingami
model <- keras_model_sequential() %>%
  # Zamiast przedstawiać, każde słowo jako 20000 elementowy wektor (jedna wartość 1 i reszta 0)
  # przedstawiamy go za pomocą wektora liczb w przestrzeni 128 wymiarowej
  layer_embedding(input_dim = 20000,
                  output_dim = 128,
                  input_length = maxlen)%>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  sentiment140_X_train,
  sentiment140_Y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)
