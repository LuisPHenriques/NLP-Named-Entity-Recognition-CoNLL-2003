# NLP-Named-Entity-Recognition-CoNLL-2003

## Implementation

This project was built using Python and jupyter notebooks, and crucial libraries like TensorFlow, Scikit-Learn, and NumPy.


## Steps involved

* Data retrieval
  1. The data was available with parquet format;
  2. The data was already separated as training, development/validation, and test datasets;

* Data Preparation
  1. Arrays had phrases, with each element being a word of each phrase, so they were constructed as one string again to be tokenized;
  2. Tokenizing enables the creation of a vocabulary for the NLP task, where each word corresponds to a unique token/id;
  3. Sequence models were used, therefore all phrases/arrays must have the same length, so all arrays were padded to the length of the longest phrase in all three data files;
  4. Although it's usual to filter certain symbols, like commas, the data already had entities to those tokens, so they were kept;

* Data Mining
  1. Instead of one-hot encoding each word, which would result in 30k features, word embedding was the approach chosen, and this way the model can learn relations between words (like the meanings we give them);
  2. While embedding the tokens, a mask was added to the model's pipeline so that the model doesn't consider the padded tokens to compute the overall loss (the cost) of the model;
  3. After adding an embedding layer, LSTM layers were chosen to be used given their particular capability of retrieving long-term data from a long sequence using its cell gate;
  4. Given the neural network should return a category for each input (token), a many-to-many architecture was employed, and so each LSTM layer returns to the following LSTM layer 124 outputs (sequence length);
  5. Stacked on top of the LSTM layers were the fully-connected (Dense) layers, the usual suspects when it comes to the normal neural networks;
  6. From what I saw, adding a TimeDistributed layer to the output layer enables the model to return more than one value, altough I didn't tested yet if that's really the case;
  7. Being a multi-label and multi-class classification problem, the output layer has as many nodes as the number of entities + 1 (no entity), therefore 9 nodes.
  8. The output layer has no activation function (linear) because the probabilities are numerically more accurate when linear values are given to the loss function, and then setting from_logits=True;
  9. There're two possible loss functions, Categorical Cross-entropy and Sparse Categorical Cross-entropy. I didn't one-hot encoded the targets, so the latter was used with the uppermentioned argument;

* Evaluation
  1. After training the model and balancing between high bias and high variance with the development set, evaluation is made with the available test dataset;
  2. The output of the model is a 3D array with values from each linear regression present on the output layer, so to these predictions is applied seperately a softmax activation function to retrieve probabilities;
  3. Having the probabilities, the outputs associated with the padded values are eliminated from the output, and then the probabilities are converted to the categoory associated with the index of the highest value.
  4. After that, the array is flatten into a 1D numpy array, so that it can be given to the confusion matrix and classification report functions from scikit-learn. The targets are also flatten for the same reason.

## Future works

  1. Implementing a Transformer network as it uses attention mechanisms to identify long-term relations like an LSTM, however a transformer can do it in parallel, while LSTM reads a sequence sequentially;
  2. Implementing convolutional layers in order to see if they can find strong filters/features while making the network lighter for the LSTM layers;
  3. Using POS (Parts-of-Speech) and chunk tags as features for the models to better understand the phrase's structure;
