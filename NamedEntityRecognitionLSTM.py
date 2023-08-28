import numpy as np
import tensorflow as tf


class NamedEntityRecognition_LSTM:
    def __init__(self, many_to_many = True, vocabulary_size = None, embedding_dimension = 64, lstm_hidden_nodes = [16], dense_hidden_nodes = [16], dense_hidden_activation = 'relu', 
                 lstm_hidden_activation = 'tanh', recurrent_activation = 'sigmoid', output_nodes = 1, output_activation = 'linear', dense_weights_initializer = 'he_normal', 
                 lstm_weighs_initializer = 'glorot_uniform', bias_initilizer = 'zeros', bidirectional_lstm = False, batch_normalization = True, layer_normalization = True, optimizer = 'adam', 
                 amsgrad_adam_mod = False, learning_rate = 0.001, learning_rate_decay = False, regularizer = 'L2', regularizer_lambda = 0.0, dropout_rate = 0.0, epochs = 40, 
                 mini_batch_size = 128, verbose = 0, loss_function = 'sparse_categorical_cross_entropy', metrics = [None]):
        
        """
        Initialize the DeepRegression_TimeSeriesLSTM model.

        Parameters:
            many_to_many (bool): Whether the LSTM is many-to-many or many-to-one.
            vocabulary_size (int): Number of words in the tokenizer's dictionary
            embedding_dimension (int): number of embeddings used in the Embedding layer.
            lstm_hidden_nodes (list): List of integers, number of LSTM units for each layer.
            dense_hidden_nodes (list): List of integers, number of dense units for each layer.
            dense_hidden_activation (str): Activation function for dense layers.
            lstm_hidden_activation (str): Activation function for LSTM layers.
            recurrent_activation (str): Activation function for recurrent connections in LSTM.
            output_nodes (int): Number of output nodes in the output layer.
            output_activation (str): Activation function for the output layer.
            dense_weights_initializer (str): Weight initializer for dense layers.
            lstm_weighs_initializer (str): Weight initializer for LSTM layers.
            bias_initilizer (str): Bias initializer for all layers.
            bidirectional_lstm (bool): Whether to use bidirectional LSTM layers.
            batch_normalization (bool): Whether to use batch normalization.
            layer_normalization (bool): Whether to use layer normalization.
            optimizer (str): Optimizer for model training.
            amsgrad_adam_mod (bool): Whether to use AMSGrad modification for Adam optimizer.
            learning_rate (float): Initial learning rate.
            learning_rate_decay (bool): Whether to apply learning rate decay during training.
            regularizer (str): Regularization method.
            regularizer_lambda (float): Regularization strength.
            dropout_rate (float): Dropout rate for dropout layers.
            epochs (int): Number of training epochs.
            mini_batch_size (int): Size of mini-batches during training.
            verbose (int): Verbosity level during training.
            loss_function (str): Loss function for model training.
            metrics (list): List of additional metrics to monitor during training.
        """
        
        self.many_to_many = many_to_many
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.lstm_hidden_nodes = lstm_hidden_nodes
        self.dense_hidden_nodes = dense_hidden_nodes
        self.dense_hidden_activation = dense_hidden_activation
        self.lstm_hidden_activation = lstm_hidden_activation
        self.recurrent_activation = recurrent_activation
        self.output_nodes = output_nodes
        self.output_activation = output_activation
        self.dense_weights_initializer = dense_weights_initializer
        self.lstm_weighs_initializer = lstm_weighs_initializer
        self.bias_initializer = bias_initilizer
        self.bidirectional_lstm = bidirectional_lstm
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.optimizer = optimizer
        self.amsgrad_adam_mod = amsgrad_adam_mod
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.regularizer = regularizer
        self.regularizer_lambda = regularizer_lambda
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.verbose = verbose
        self.loss_function = loss_function
        self.metrics = metrics


    @staticmethod
    def _set_optimizer(optimizer_name, learning_rate, amsgrad_bol):

        """
        Set and configure an optimizer for training.

        Parameters:
            optimizer_name (str): Name of the optimizer.
            learning_rate (float): Learning rate for the optimizer.
            amsgrad_bol (bool): Whether to use AMSGrad modification for Adam optimizer.

        Returns:
            optimizer (tf.keras.optimizers.Optimizer): Configured optimizer.
        """

        if optimizer_name == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-07)
        
        elif optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=amsgrad_bol)

        elif optimizer_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

        elif optimizer_name == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

        return optimizer


    @staticmethod
    def _learning_rate_decay(learning_rate, with_decay):

        """
        Generate a learning rate schedule for decay.

        Parameters:
            learning_rate (float): Initial learning rate.
            with_decay (bool): Whether to apply learning rate decay.

        Returns:
            lr_schedule (list): Learning rate schedule as a list of callbacks.
        """

        if with_decay:
            lr_schedule = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate / (1 + 0.96 * epoch))]

        else:
            lr_schedule = []

        return lr_schedule


    @staticmethod
    def _set_loss_function(loss_name):

        """
        Set the appropriate loss function based on the provided name.

        Parameters:
            loss_name (str): Name of the loss function.

        Returns:
            loss_func (tf.keras.losses.Loss): Configured loss function.
        """

        if loss_name == 'sparse_categorical_cross_entropy':
            loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        else:
            loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                
        return loss_func


    @staticmethod
    def _set_regularizer(regularizer_name, lambdaa):

        """
        Set a regularization term based on the provided name.

        Parameters:
            regularizer_name (str): Name of the regularization method.
            lambdaa (float): Regularization strength.

        Returns:
            regularizer (tf.keras.regularizers.Regularizer or None): Configured regularization term or None if not needed.
        """

        if regularizer_name == 'L2':
            regularizer = tf.keras.regularizers.L2(lambdaa)

        else:
            regularizer = None

        return regularizer


    def CompileTrainModel(self, tokenizer, sequence_length, model_name, final_model=False, show_output=True):

        """
            Compile and train a LSTM model for Time Series forecasting.
            
            Parameters:
                X_train (numpy array): Training input data.
                Y_train (numpy array): Training target data.
                X_valid (numpy array): Validation input data.
                Y_valid (numpy array): Validation target data.
                model_name (str): Name of the model.
                final_model (bool): Whether to return the final trained model.
                show_output (bool): Whether to show the model summary.
                
            Returns:
                model (TensorFlow model): Trained model.
                history (dic): Dictonary with training and validation loss values for each epoch.
        """

        # Initialize a Sequential model
        model=tf.keras.models.Sequential(name = model_name)
    
        # Weights regularizer (L2)
        regularizer = NamedEntityRecognition_LSTM._set_regularizer(self.regularizer, self.regularizer_lambda)

        # Get the number of LSTM and Dense layers
        num_lstm_layers = len(self.lstm_hidden_nodes)
        num_dense_layers = len(self.dense_hidden_nodes)

        # Construct Embedding layer
        model.add(tf.keras.layers.Embedding(
                                            input_dim = self.vocabulary_size + 1, 
                                            output_dim = self.embedding_dimension, 
                                            input_length = sequence_length,
                                            mask_zero = True
                                            )
                )
        
        # Construct LSTM layers
        for layer in range(num_lstm_layers):

            if self.bidirectional_lstm:
                # Add a bidirectional LSTM layer
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                                                                units = self.lstm_hidden_nodes[layer],
                                                                                kernel_initializer = self.lstm_weighs_initializer,
                                                                                bias_initializer = self.bias_initializer,
                                                                                kernel_regularizer = regularizer,
                                                                                activation=self.lstm_hidden_activation,
                                                                                recurrent_activation = self.recurrent_activation,
                                                                                return_sequences = self.many_to_many
                                                                                ), name = "Bidirectional-LSTM-Layer-" + str(layer + 1)
                                                            )
                            )
                    
            else:
                # Add a regular LSTM layer
                model.add(tf.keras.layers.LSTM(
                                                units = self.lstm_hidden_nodes[layer],
                                                name = "LSTM-Layer-" + str(layer + 1),
                                                kernel_initializer = self.lstm_weighs_initializer,
                                                bias_initializer = self.bias_initializer,
                                                kernel_regularizer = regularizer,
                                                activation = self.lstm_hidden_activation,
                                                recurrent_activation = self.recurrent_activation,
                                                return_sequences = self.many_to_many
                                                )
                        )
                    
            # Add layer normalization
            if self.layer_normalization:
                model.add(tf.keras.layers.LayerNormalization())

        # Construct Dense layers
        for layer in range(num_dense_layers):

            if layer == 0:
                # Add the first Dense layer
                model.add(tf.keras.layers.Dense(
                                                units = self.dense_hidden_nodes[layer],
                                                name = "Dense-Layer-" + str(layer + 1),
                                                kernel_initializer = self.dense_weights_initializer,
                                                bias_initializer = self.bias_initializer,
                                                kernel_regularizer = regularizer,
                                                activation = self.dense_hidden_activation
                                                )
                            ) 
                
            else:
                # Add subsequent Dense layers with optional BatchNormalization and Dropout
                if self.batch_normalization:
                    model.add(tf.keras.layers.BatchNormalization())

                if self.dropout_rate > 0.0:
                    model.add(tf.keras.layers.Dropout(self.dropout_rate))

                model.add(tf.keras.layers.Dense(
                                                units = self.dense_hidden_nodes[layer],
                                                name = "Dense-Layer-" + str(layer + 1),
                                                kernel_initializer = self.dense_weights_initializer,
                                                bias_initializer = self.bias_initializer,
                                                kernel_regularizer = regularizer,
                                                activation = self.dense_hidden_activation
                                                )
                            ) 

        # Add BatchNormalization and Dropout to the output layer
        if self.batch_normalization:
                    model.add(tf.keras.layers.BatchNormalization())

        if self.dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Add the output layer
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units = self.output_nodes, activation = self.output_activation), name = "Output-Layer"))

        # Set optimizer, loss function, and learning rate schedule
        optimizer = NamedEntityRecognition_LSTM._set_optimizer(self.optimizer, self.learning_rate, self.amsgrad_adam_mod)
        loss_function = NamedEntityRecognition_LSTM._set_loss_function(self.loss_function)
        lr_schedule = NamedEntityRecognition_LSTM._learning_rate_decay(self.learning_rate, self.learning_rate_decay)
        
        # Compile the model
        model.compile(loss = loss_function, optimizer = optimizer, metrics = self.metrics)
        
        # Show the model summary if required
        if show_output:
            print("\n******************************************************")
            model.summary()
        
        # Train the model
        history = model.fit(
                            x = tokenizer.train,
                            epochs = self.epochs,
                            verbose = self.verbose,
                            validation_data = tokenizer.dev,
                            callbacks = lr_schedule,
                            use_multiprocessing = True,
                            workers = 6
                            )
        
        # Return the model and history if final_model=True, else return history
        if final_model:
            return model, history
        else:
            return history
