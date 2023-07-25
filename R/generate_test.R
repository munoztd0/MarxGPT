


library(torch)
library(text)
library(keras)

# Check if CUDA is available and set the device accordingly

if (cuda_is_available()) {
  device <- torch_device("cuda")
  cat("CUDA is available. Using GPU.\n")
} else {
  device <- torch_device("cpu")
  cat("CUDA is not available. Using CPU.\n")
}

# Load the South Park lines corpus
shakespeare_text <- readLines("../data/southpark_corpus.txt")

# Convert the text to a single character string
shakespeare_text <- paste(shakespeare_text, collapse=" ")


tokenizer <- text_tokenizer(num_words = 1000)  # Set num_words to an appropriate value
tokenizer$fit_on_texts(shakespeare_text)
sequences <- tokenizer$texts_to_sequences(shakespeare_text)

seq_length <- 10  # Set the sequence length (window size)
X <- list()
y <- list()



for (i in 1:(length(sequences) - seq_length)) {
  X[[i]] <- torch_tensor(as.numeric(unlist(sequences[i:(i + seq_length - 1)])), device="cpu")
  y[[i]] <- torch_tensor(as.numeric(unlist(sequences[i + seq_length])), device="cpu")
}

library(rTorch)

# Convert the PyTorch tensors to NumPy arrays
X <- lapply(X, function(x) if (is_tensor(x)) as.matrix(x$numpy()) else x)
y <- lapply(y, function(x) if (is_tensor(x)) as.matrix(x$numpy()) else x)



# Pad the sequences to a common length
X <- pad_sequences(X, value=0, maxlen=seq_length)


X <- torch_stack(X)

y <- torch_stack(y)


input_size <- tokenizer$word_index_length()
hidden_size <- 128
output_size <- tokenizer$word_index_length()

rnn_model <- torch_module("rnn", list(
  "rnn" = torch_rnn(input_size, hidden_size, batch_first = TRUE),
  "fc" = torch_linear(hidden_size, output_size)
))

loss_fn <- torch_cross_entropy_loss()
optimizer <- torch_optimizer_adam(rnn_model$parameters(), lr = 0.01)

num_epochs <- 100
for (epoch in 1:num_epochs) {
  optimizer$zero_grad()

  # Forward pass
  outputs <- rnn_model(X)

  # Calculate loss
  loss <- loss_fn(outputs$permute(0, 2, 1), y)

  # Backward pass and optimize
  loss$backward()
  optimizer$step()

  cat(sprintf("Epoch [%d/%d], Loss: %.4f\n", epoch, num_epochs, loss$item()))
}


# Choose a random starting point from the input data
start_index <- sample(1:(length(sequences) - seq_length), 1)

# Generate text from the RNN model
generated_text <- X[start_index,,]
generated_text <- torch_reshape(generated_text, c(1, seq_length))

num_words_to_generate <- 50
for (i in 1:num_words_to_generate) {
  output_probs <- rnn_model(generated_text)
  next_word_index <- as.numeric(torch_argmax(output_probs$detach(), dim = 2))
  generated_text <- torch_cat(generated_text, next_word_index, dim = 2)
}

# Convert the generated sequence back to text
generated_text <- as.array(generated_text)[1,]
generated_text <- tokenizer$sequences_to_texts(generated_text)
cat(generated_text)






















# Tokenize the text into bigrams
bigrams <- substr(text, 1, nchar(text)-1) |>
    paste0(substr(text, 2, nchar(text)))


# Create a table of bigram frequencies
freq_table <- table(bigrams)

# Define a function to generate text
generate_text <- function(n, freq_table) {
  # Choose a random starting bigram
  current_bigram <- sample(names(freq_table), 1)
  # Initialize the generated text
  generated_text <- current_bigram
  # Generate n-1 more bigrams
  for (i in 2:n) {
    # Get the frequencies of all bigrams starting with the current bigram
    next_bigrams <- names(freq_table[grep(paste0("^", current_bigram), names(freq_table))])
    # If there are no bigrams starting with the current bigram, choose a new random bigram
    if (length(next_bigrams) == 0) {
      current_bigram <- sample(names(freq_table), 1)
      generated_text <- paste(generated_text, current_bigram, sep="")
    } else {
      # Choose the next bigram based on its frequency
      next_bigram <- sample(next_bigrams, 1, prob=freq_table[next_bigrams]/sum(freq_table[next_bigrams]))
      current_bigram <- substr(next_bigram, nchar(next_bigram)-1, nchar(next_bigram))
      generated_text <- paste(generated_text, current_bigram, sep="")
    }
  }
  return(generated_text)
}

# Generate 100 bigrams of text
generated_text <- generate_text(10, freq_table)

# Print the generated text
print(generated_text)
