from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset


class DataTokens():

    """
    A class to handle tokenization and preprocessing of text data for NLP tasks.

    Attributes:
        train (DataFrame): Training data.
        dev (DataFrame): Development data.
        test (DataFrame): Test data.
        phrases_column (str): Name of the column containing text phrases.
        entities_column (str): Name of the column containing NER entities.
        num_words (int): Maximum number of words to keep.
        words_in_list (bool): Whether phrases are stored as lists of words.
        custom_filter (str): Characters to be filtered out during tokenization.
    """

    def __init__(self, train_df, dev_df, test_df, phrases_column, entities_column, num_words = 100000, words_in_list = True, 
                 custom_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', mini_batch_size = 256):

        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.phrases_column = phrases_column
        self.entities_column = entities_column
        self.num_words = num_words
        self.words_in_list = words_in_list
        self.custom_filter = custom_filter
        self.mini_batch_size = mini_batch_size


    def __repr__(self):
        repr_str = (
            f"Tokenizer Object\n"
            f"=================\n"
            f"Shape of preprocessed training tokens:      {self.X_train.shape}\n"
            f"Shape of preprocessed development tokens:   {self.X_dev.shape}\n"
            f"Shape of preprocessed testing tokens:       {self.X_test.shape}\n\n"
            f"Shape of preprocessed training NER tags:    {self.Y_train.shape}\n"
            f"Shape of preprocessed development NER tags: {self.Y_dev.shape}\n"
            f"Shape of preprocessed testing NER tags:     {self.Y_test.shape}\n\n"
            f"Tokens representation: {self.X_train[0]}\n"
            f"NER tags representation: {self.Y_train[0]}"
        )
        return repr_str


    def TokenizePhrases(self):

        """
        Tokenizes the input phrases using the Tokenizer class.

        Returns:
            train_tokens (list): Tokenized training phrases.
            dev_tokens (list): Tokenized development phrases.
            test_tokens (list): Tokenized test phrases.
        """

        if self.words_in_list:
            self.train_df[self.phrases_column] = self.train_df[self.phrases_column].apply(lambda phrase: " ".join(phrase))
            self.dev_df[self.phrases_column] = self.dev_df[self.phrases_column].apply(lambda phrase: " ".join(phrase))
            self.test_df[self.phrases_column] = self.test_df[self.phrases_column].apply(lambda phrase: " ".join(phrase))

        # Initialize the Tokenizer class
        tokenizer = Tokenizer(num_words = self.num_words, oov_token="<OOV>", filters = self.custom_filter)

        # Tokenize the input sentences
        tokenizer.fit_on_texts(self.train_df[self.phrases_column])

        # Get the word index dictionary
        word_index = tokenizer.word_index
        self.word_index = word_index

        # Generate list of token sequences
        train_tokens = tokenizer.texts_to_sequences(self.train_df[self.phrases_column])
        dev_tokens = tokenizer.texts_to_sequences(self.dev_df[self.phrases_column])
        test_tokens = tokenizer.texts_to_sequences(self.test_df[self.phrases_column])

        return train_tokens, dev_tokens, test_tokens
    

    def PreprocessTokens(self):

        """
        Preprocesses tokenized phrases by padding them to a uniform length.

        Returns:
            train_tokens (numpy.ndarray): Padded training tokens.
            dev_tokens (numpy.ndarray): Padded development tokens.
            test_tokens (numpy.ndarray): Padded test tokens.
        """

        train_tokens, dev_tokens, test_tokens = self.TokenizePhrases()

        # Find the maximum length of the sequences so no conflict occurs when validating or testing the models
        maxlen = max([len(max(train_tokens, key=len)), len(max(dev_tokens, key=len)), len(max(test_tokens, key=len))])
        self.maxlen = maxlen

        # Pad the sequences to a uniform length
        train_tokens = pad_sequences(train_tokens, padding = 'post', maxlen = self.maxlen)
        dev_tokens = pad_sequences(dev_tokens, padding = 'post', maxlen = self.maxlen)  
        test_tokens = pad_sequences(test_tokens, padding = 'post', maxlen = self.maxlen)

        self.X_train = train_tokens
        self.X_dev = dev_tokens
        self.X_test = test_tokens
    

    def PreprocessTags(self):

        """
        Preprocesses NER tags by padding them to a uniform length.

        Returns:
            train_ner (numpy.ndarray): Padded training NER tags.
            dev_ner (numpy.ndarray): Padded development NER tags.
            test_ner (numpy.ndarray): Padded test NER tags.
        """

        # Get the NER tags from the pandas dataframes in the appropriate format
        train_ner = [seq.tolist() for seq in self.train_df[self.entities_column].tolist()]
        dev_ner = [seq.tolist() for seq in self.dev_df[self.entities_column].tolist()]
        test_ner = [seq.tolist() for seq in self.test_df[self.entities_column].tolist()]

        # Pad the sequences to a uniform length
        train_ner = pad_sequences(train_ner, padding = 'post', maxlen = self.maxlen)
        dev_ner = pad_sequences(dev_ner, padding = 'post', maxlen = self.maxlen)  
        test_ner = pad_sequences(test_ner, padding = 'post', maxlen = self.maxlen)

        self.Y_train = train_ner
        self.Y_dev = dev_ner
        self.Y_test = test_ner


    def make_dataset_train(self, x_data, y_data):

        """
        Creates a TensorFlow Dataset with paired training tokens and NER tags.

        Args:
            x_data (numpy.ndarray): Training tokens data.
            y_data (numpy.ndarray): Training NER tags data.

        Returns:
            dataset (tf.data.Dataset): TensorFlow Dataset containing paired tokens and tags.
        """

        dataset = Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.shuffle(buffer_size = 1000, seed=0)
        dataset = dataset.batch(self.mini_batch_size)
        dataset = dataset.prefetch(1)

        return dataset
    

    def make_dataset(self, x_data, y_data):

        """
        Creates a TensorFlow Dataset with paired training tokens and NER tags.

        Args:
            x_data (numpy.ndarray): Training tokens data.
            y_data (numpy.ndarray): Training NER tags data.

        Returns:
            dataset (tf.data.Dataset): TensorFlow Dataset containing paired tokens and tags.
        """

        dataset = Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.batch(self.mini_batch_size)
        dataset = dataset.prefetch(1)

        return dataset
     

    @property
    def train(self):

        """
        Property for the training dataset.
        """
        return self.make_dataset_train(self.X_train, self.Y_train)

    @property 
    def dev(self):
            
        """
        Property for the development/validation dataset.
        """
        return self.make_dataset(self.X_dev, self.Y_dev)

    @property 
    def test(self):
            
        """
        Property for the test dataset.
        """
        return self.make_dataset(self.X_test, self.Y_test)
    