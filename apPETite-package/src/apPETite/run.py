import tensorflow as tf
import numpy as np
import os
import random

# CONSTRUCT_DIWV()
# Returns a dictionary that maps pairs of amino acids to their instability score according to ProtParam's Instability Index.
# The csv file we used for our index is included, but you are welcome to use your own file.
def construct_diwv():
  diwv_dict = {}
  matrix = []
  AMINO_ACIDS = ['W', 'C', 'M', 'H', 'Y', 'F', 'Q', 'N', 'I', 'R', 'D', 'P', 'T',
               'K', 'E', 'V', 'S', 'G', 'A', 'L']

  # diwv.csv is the matrix of instability values found in
  # Guruprasad, K., Reddy, B.V.B. and Pandit, M.W. (1990)
  with open("./data/diwv.csv", "r") as f:
    for line in f.readlines():
      scores = line.split(",")
      matrix.append([float(s) for s in scores])
  for i in range(20):
    subdict = {}
    for j in range(20):
      subdict[AMINO_ACIDS[j]] = matrix[i][j]
    diwv_dict[AMINO_ACIDS[i]] = subdict
  return diwv_dict

# PASSES_FILTERS()
# Filter based on sequence length and stability.
# You are welcome to add any other filter or parameter that you are analyzing for your specific protein.
def passes_filters(seq, min_len=270, max_len=310,
                   max_instability_index=40,
                   max_conserved_residue_penalty=20):
  
  diwv_dict = construct_diwv()

  # controls sequence length
  if len(seq) <= 270 or len(seq) >= 310:
    return False
  
  # Controls sequence stability (ProtParam)
  score = 0
  for i in range(len(seq)-1):
    score += diwv_dict[seq[i]][seq[i+1]]
    score = 10.0/len(seq) * score
  if score > 40.0:
    return False
  else: 
    return True

# RUN_RNN()
# Recurrent neural network that returns novel PETase sequences that pass filters created in passes_filters().
# Text profile included is our training set, but this can be replaced by your own training set of sequences.
# Sizes for each parameter can be adjusted to your needs.
def run_RNN(seq_length=100, batch_size=2, BUFFER_SIZE=10000, epochs=30,
            embedding_dim=256, rnn_units=1024, num_generate=290,
            temperature=0.1, seed=u"MNFPRASRLMQAAVL",
            path_to_file="./data/sequences-final.txt", num_seqs = 5):
  
  # the full string of all sequences
  text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
  text = text.replace('\n', '')
  text = text.replace('\r', '')
  text = text.replace(' ', '')
  print ('Length of text: {} characters'.format(len(text)))

  # a collection of the unique letters in the file, should be 22
  vocab = sorted(set(text))
  print('{} unique characters'.format(len(vocab)))

  # unique mapping from letters to integers
  char2idx = {u:i for i, u in enumerate(vocab)}

  # unique mapping from integers to letters
  idx2char = np.array(vocab)

  # the integer representation of the string
  text_as_int = np.array([char2idx[c] for c in text])
  print('{} ---map---> {}'.format(text[:13], text_as_int[:13]))

  examples_per_epoch = len(text)//(seq_length+1)

  inputs = []
  outputs = []
  # cuts the input file into slices of length==seq_length
  # each slice is an input
  # the same slice but one letter shifted to the right is its desired output
  for i in range(seq_length, len(text), seq_length):
    inputs.append(text_as_int[i-seq_length:i])
    outputs.append(text_as_int[i-seq_length+1:i+1])
  inputs = inputs[:-1]
  outputs = outputs[:-1]

  # groups the data as (input_i, output_i) pairs, so shuffling them preserves
  # their correspondance
  dataset = list(zip(inputs, outputs))
  random.shuffle(dataset)
  dataset = list(zip(*dataset))
  X, y = np.array(dataset[0]), np.array(dataset[1])
  print(X.shape)
  print(y.shape)

  # batch size must be able to divide into the number of training examples
  batch_size = 2
  vocab_size = len(vocab)

  def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape = [batch_size, seq_length]),
      tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'), 
      # tf.keras.layers.Flatten(batch_input_shape = [batch_size, seq_length]),
      # tf.keras.layers.Reshape(batch_size, seq_length), 
      tf.keras.layers.Dense(vocab_size, activation = 'relu')
    ])
    return model

  model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)
  
  model.summary()

  def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

  model.compile(optimizer='adam', loss=loss)

  # Setup checkpoints
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  # Train the model
  history = model.fit(X, y, epochs=epochs, callbacks=[checkpoint_callback], batch_size=batch_size)

  # Load the latest model
  tf.train.latest_checkpoint(checkpoint_dir)

  # Prepare for sequence generation
  model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  model.build(tf.TensorShape([1, None]))

  # Returns one string generated by the RNN <model>, based on a short seed
  # given as <start_string>
  def generate_text(model, start_string):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        
        # samples the alphabet based on <predictions> interpreted as a vector
        # of probabilities per letter
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


  valid_seqs = []
  while len(valid_seqs) < num_seqs:
    blabble = generate_text(model, seed)
    # print(blabble)
    # print(len(valid_seqs))
    # seq = blabble.split("\n") #[0]
    seq = blabble
    # print(len(seq))
    if not passes_filters(seq): #seq.strip()
      continue
    else:
      valid_seqs.append(seq)
      
  return valid_seqs