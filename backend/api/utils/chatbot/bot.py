# import json
# import re
# import numpy as np
# import nltk
# import pickle
# import random
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.models import load_model
# from matplotlib import pyplot as plt
#
# BASE_DIR = 'api/utils/chatbot/'
# MODEL_DIR = f'{BASE_DIR}saved_models/'
# HELPER_DIR = f'{BASE_DIR}helper/'
# DATASET_DIR = f'{BASE_DIR}dataset/'
#
#
# class ChatBot:
#     """
#     A class for creating a chatbot.
#     Attributes:
#         intents (dict): A dictionary of intents, where each intent is a dictionary with the following keys:
#             - patterns (list): A list of patterns that the intent matches.
#             - tag (str): The name of the intent.
#         all_words (list): A list of all the words in the intents.
#         classes (list): A list of all the intent tags.
#         dataset (list): A list of tuples, where each tuple consists of a list of tokenized words and the intent tag.
#         language (str): Language of the chatbot.
#         mode (str): Chatbot running mode.
#             - development: Debugging logs will be enabled.
#             - production: Debugging logs will be disabled.
#
#     Methods:
#         preprocess(): Preprocesses the intents, creating the all_words and classes lists.
#         create_pkl_files(): Creates the `all_words.pkl` and `classes.pkl` files.
#         gen_graph(hist, title): Plots the accuracy and loss graphs for the training history.
#         get_train_set(): Creates a training set from the dataset list.
#         xy_split(train_set): Splits the training set into the x and y components.
#         create_model(input_shape, output_shape): Creates a Keras model for training.
#         train(epochs=200): Trains the Keras model.
#         clean_up(sentence): Cleans up a sentence, removing punctuation and stemming the words.
#         bag_of_words(sentence): Creates a bag-of-words representation of a sentence.
#         get_predictions(sentence): Gets the predictions for a sentence.
#     """
#     IGNORED_LETTERS = ',.?!/-_'
#
#     def __init__(self, language='english', json_file='intents.json', mode='production'):
#         """
#         Initialize the ChatBot.
#
#         Args:
#             language (str): The language of the chatbot. Default is 'english'.
#             json_file (str): The path to the JSON file containing the intents. Default is './dataset/intents.json'.
#             mode (str): The running mode of the chatbot. Default is 'development'.
#         """
#         # ! TODO fix this
#         if language == 'en':
#             language = 'english'
#         else:
#             language = 'sinhala'
#
#         self.language = language
#         json_file_path = f'{DATASET_DIR}intents_{self.language}.json'
#         self.intents = json.loads(open(json_file_path, encoding='utf-8').read())
#         self.lemmatizer = WordNetLemmatizer()
#         self.stemmer = SnowballStemmer(language='english')
#         self.all_words = []
#         self.classes = []
#         self.dataset = []
#         self.mode = mode  # development or production mode
#         self.model_path = f'{MODEL_DIR}chat_bot_model_{self.language}'
#         self.helper_file_path = HELPER_DIR
#         self.all_words_file_path = f'{self.helper_file_path}all_words_{self.language}.pkl'
#         self.classes_file_path = f'{self.helper_file_path}classes_{self.language}.pkl'
#
#     def preprocess(self):
#         """
#         Preprocess the intents data by tokenizing words and creating the dataset.
#         """
#         for intent in self.intents['intents']:
#             for pattern in intent['patterns']:
#                 # Remove punctuation
#                 pattern = re.sub(r'[^\w\s]', '', pattern)
#                 # Tokenize
#                 tokenized_words = nltk.word_tokenize(pattern)
#
#                 self.all_words.extend(tokenized_words)
#                 self.dataset.append((tokenized_words, intent['tag']))
#
#                 if intent['tag'] not in self.classes:
#                     self.classes.append(intent['tag'])
#
#         if self.language == 'english':
#             self.all_words = [self.lemmatizer.lemmatize(word) for word in self.all_words if word not in self.IGNORED_LETTERS]
#         else:
#             # If Sinhala language
#             self.all_words = [word for word in self.all_words if word not in self.IGNORED_LETTERS]
#
#         self.all_words = sorted(set(self.all_words))
#         self.classes = sorted(set(self.classes))
#
#     def create_pkl_files(self):
#         """
#         Create pickle files for all_words and classes.
#         """
#         pickle.dump(self.all_words, open(self.all_words_file_path, 'wb'))
#         pickle.dump(self.classes, open(self.classes_file_path, 'wb'))
#
#     def gen_graph(self, hist, title):
#         """
#         Generate a graph for model accuracy and loss.
#
#         Args:
#             hist: The history object returned from model.fit().
#             title (str): The title of the graph.
#         """
#         plt.plot(hist.history['accuracy'])
#         plt.title('Model Accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Epoch')
#         plt.show()
#
#         plt.plot(hist.history['loss'])
#         plt.title('Model Loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.show()
#
#     def get_train_set(self):
#         """
#         Generate the training dataset.
#
#         Returns:
#             np.array: The training dataset.
#         """
#         train_set = []
#         enc_arr = [0] * len(self.classes)
#
#         for data in self.dataset:
#             bag = []
#             patterns = data[0]
#
#             if self.language == 'english':
#                 # Lemmatize words with at least two letters, otherwise neglect the word
#                 patterns = [self.lemmatizer.lemmatize(word.lower()) for word in patterns if len(word) > 1]
#             else:
#                 # Get individual words into list if word has at least 2 letters
#                 patterns = [word.lower() for word in patterns if len(word) > 1]
#
#             if self.mode == 'development':
#                 print('Patterns:', patterns)
#
#             for word in self.all_words:
#                 if word in patterns:
#                     bag.append(1)
#                 else:
#                     bag.append(0)
#
#             tmp_enc_arr = list(enc_arr)
#             tmp_enc_arr[self.classes.index(data[1])] = 1
#             train_set.append([bag, tmp_enc_arr])
#             random.shuffle(train_set)
#         print(train_set)
#         return np.array(train_set)
#
#     def xy_split(self, train_set):
#         """
#         Split the train set into input and output.
#
#         Args:
#             train_set: The training dataset.
#
#         Returns:
#             tuple: The input and output arrays.
#         """
#         x_train = list(train_set[:, 0])
#         y_train = list(train_set[:, 1])
#         return x_train, y_train
#
#     def create_model(self, input_shape, output_shape):
#         """
#         Create the model architecture.
#
#         Args:
#             input_shape: The shape of the input data.
#             output_shape: The shape of the output data.
#
#         Returns:
#             Sequential: The compiled model.
#         """
#         model = Sequential()
#         model.add(Dense(6, input_shape=input_shape, activation="relu"))
#         model.add(BatchNormalization())
#         model.add(Dense(16, input_shape=input_shape, activation="relu"))
#         model.add(BatchNormalization())
#         model.add(Dense(64, activation="relu"))
#         model.add(BatchNormalization())
#         model.add(Dense(output_shape, activation="softmax"))
#
#         adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
#
#         model.compile(loss='categorical_crossentropy',
#                       optimizer=adam,
#                       metrics=["accuracy"])
#         if self.mode == 'development':
#             print(model.summary())
#         return model
#
#     def train(self, epochs=200):
#         """
#         Train the model.
#
#         Args:
#             epochs (int): The number of epochs to train the model for.
#         """
#         self.preprocess()
#         self.create_pkl_files()
#         train_set = self.get_train_set()
#
#         x_train, y_train = self.xy_split(train_set)
#         input_shape = (len(x_train[0]),)
#         output_shape = len(y_train[0])
#
#         model = self.create_model(input_shape, output_shape)
#         hist = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1)
#         if self.mode == 'development':
#             title = f'Accuracy and Loss Curve of Neural Net for {self.language} Model'
#             self.gen_graph(hist, title)
#         model.save(self.model_path)
#
#     def clean_up(self, sentence):
#         """
#         Clean up the sentence by tokenizing and stemming words.
#
#         Args:
#             sentence (str): The sentence to clean up.
#
#         Returns:
#             list: The cleaned up words.
#         """
#         # Remove punctuation
#         sentence = re.sub(r'[^\w\s]', '', sentence)
#         tokenized = nltk.word_tokenize(sentence)
#
#         if self.language == 'english':
#             cleaned_words_arr = [self.lemmatizer.lemmatize(word) for word in tokenized]
#         else:
#             cleaned_words_arr = [word for word in tokenized]
#         return cleaned_words_arr
#
#     def bag_of_words(self, sentence):
#         """
#         Convert a sentence to a bag of words representation.
#
#         Args:
#             sentence (str): The sentence to convert.
#
#         Returns:
#             np.array: The bag of words representation.
#         """
#         all_words = pickle.load(open(self.all_words_file_path, 'rb'))
#
#         cleaned_words = self.clean_up(sentence)
#         bag = [0] * len(all_words)
#
#         if self.mode == 'development':
#             print('All word file:', self.all_words_file_path)
#             print('Cleaned sentence:', cleaned_words)
#
#         for word in cleaned_words:
#             for idx, w in enumerate(all_words):
#                 if word == w:
#                     bag[idx] = 1
#         return np.array(bag)
#
#     def get_predictions(self, sentence):
#         """
#         Get the intent predictions for a given sentence.
#
#         Args:
#             sentence (str): The input sentence.
#
#         Returns:
#             list: A list of intent predictions with probabilities.
#         """
#         bag = self.bag_of_words(sentence)
#         loaded_model = load_model(self.model_path)
#
#         predictions = loaded_model.predict(np.array([bag]))[0]
#         ERROR_THRESHOLD = 0.25
#         results = [[i, r] for i, r in enumerate(predictions) if r > ERROR_THRESHOLD]
#
#         classes = pickle.load(open(self.classes_file_path, 'rb'))
#         results.sort(key=lambda x: x[1], reverse=True)
#         return_list = []
#         for result in results:
#             return_list.append(
#                 {
#                     'intent': classes[result[0]],
#                     'probability': result[1]
#                 }
#             )
#         return return_list
#
#     def get_response(self, tag):
#         """ returns random response according to intent"""
#         for intent in self.intents['intents']:
#             if intent['tag'] == tag:
#                 responses = intent['responses']
#         response = random.choice(responses)
#         return response

#%%

import os
import json
import string
import random
import pickle
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

BASE_DIR = 'api/utils/chatbot'

class ChatBot:
    """
    A class for creating a chatbot.

    Attributes:
        intents (dict): A dictionary of intents, where each intent is a dictionary with the following keys:
            * patterns (list): A list of patterns that the intent matches.
            * tag (str): The name of the intent.
        all_words (list): A list of all the words in the intents.
        classes (list): A list of all the intent tags.
        dataset (list): A list of tuples, where each tuple consists of a list of tokenized words and the intent tag.
        language (str): Language of the chatbot
        mode (str): Chatbot running modes
                        - development : Debugging logs will be enabled
                        - production : Debugging logs will be disabled

    Methods:
        preprocess(): Preprocesses the intents, creating the all_words and classes lists.
        create_pkl_files(): Creates the `all_words.pkl` and `classes.pkl` files.
        gen_graph(hist, title): Plots the accuracy and loss graphs for the training history.
        get_train_set(): Creates a training set from the dataset list.
        xy_split(train_set): Splits the training set into the x and y components.
        create_model(input_shape, output_shape): Creates a Keras model for training.
        train(epochs=200): Trains the Keras model.
        clean_up(sentence): Cleans up a sentence, removing punctuation and stemming the words.
        bag_of_words(sentence): Creates a bag-of-words representation of a sentence.
    """
    def __init__(self, language='en'):
        self.intents = None
        self.language = language
        self.lm = WordNetLemmatizer()
        self.classes = []
        self.vocab = []
        self.training_data = []
        self.model = None
        self.mode = 'development'

    def load_intents(self):
        # Load intents data from JSON file based on the specified language
        intents_file = f'{BASE_DIR}/dataset/intents_{self.language}.json'
        with open(intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

    def prepare_data(self):
        """
        Load and preprocess intent data for training the chatbot.
            Args:
            intents_data (dict): A dictionary containing intent data with tags, patterns, and responses.
        Returns:
            None
        """
        # load intents file
        self.load_intents()
        # Get unique intent tags and sort them
        self.classes = sorted(set(intent["tag"] for intent in self.intents["intents"]))

        # Process patterns and build vocabulary and training data
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                # Tokenize pattern text into individual words
                tokens = nltk.word_tokenize(pattern)

                if self.language == 'en':
                    # Lemmatize words and convert to lowercase, excluding punctuation
                    tokens = [self.lm.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]
                else:
                    tokens = [word for word in tokens if word not in string.punctuation]

                # Extend vocabulary with processed tokens
                self.vocab.extend(tokens)
                # Store tokenized pattern and associated intent tag in training data
                self.training_data.append((tokens, intent["tag"]))

        # Remove duplicates and sort vocabulary
        self.vocab = sorted(set(self.vocab))
        # save vocab and classes
        self.save_data()

    def save_data(self):
        # Save the vocabulary and classes as pickle files
        vocab_path = f'{BASE_DIR}/helper/chatbot_{self.language}/vocab.pkl'
        classes_path = f'{BASE_DIR}/helper/chatbot_{self.language}/classes.pkl'
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(classes_path, 'wb') as f:
            pickle.dump(self.classes, f)

    def load_model_and_data(self):
        self.load_intents()
        self.vocab = self.load_pkl_file(f'{BASE_DIR}/helper/chatbot_{self.language}/vocab.pkl')
        self.classes = self.load_pkl_file(f'{BASE_DIR}/helper/chatbot_{self.language}/classes.pkl')
        self.model = load_model(f'{BASE_DIR}/saved_models/chatbot_{self.language}/model')

    def load_pkl_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def create_model(self):
        """
        Create and compile the deep learning model for intent classification.
        Returns:
            history: Training history of the model.
        """
        # Prepare training data
        x_train = []
        y_train = []

        for (tokens, intent_tag) in self.training_data:
            # Convert tokens to bag of words representation
            bag = [1 if word in tokens else 0 for word in self.vocab]

            # Prepare one-hot encoded output for the intent tag
            output = [0] * len(self.classes)
            output[self.classes.index(intent_tag)] = 1

            x_train.append(bag)
            y_train.append(output)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Define input and output shapes for the model
        i_shape = (len(x_train[0]),)
        o_shape = len(y_train[0])

        # Build the sequential model
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=i_shape, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(o_shape, activation="softmax"))

        # Define learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9
        )

        # Compile the model with Adam optimizer and categorical cross-entropy loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # Train the model and store training history
        hist = self.model.fit(x_train, y_train, epochs=200, verbose=1)
        # if self.mode == 'development':
        #     self.gen_graph(hist, 'Training/loss curve')

    def preprocess_input(self, text):
        """
        Preprocess the input text by tokenizing and lemmatizing.
        Args:
            text (str): Input text to be preprocessed.
        Returns:
            preprocessed_tokens (list): List of preprocessed tokens.
        """
        # Tokenize the input text
        tokens = nltk.word_tokenize(text)
        if self.language == 'en':
            # Lemmatize each token and convert to lowercase
            tokens = [self.lm.lemmatize(word) for word in tokens]
        return tokens

    def predict_intent(self, text):
        """
        Predict the intent(s) of the given text.
        Args:
            text (str): Input text for which intent(s) are to be predicted.
        Returns:
            predicted_intents (list): List of predicted intents based on the text.
        """
        # Preprocess the input text to get tokens
        tokens = self.preprocess_input(text)
        # Create a bag of words representation based on vocabulary
        bag = [1 if word in tokens else 0 for word in self.vocab]
        # Make predictions using the model
        predictions = self.model.predict(np.array([bag]))[0]
        # Set a threshold for prediction confidence
        threshold = 0.2
        # Select intents with prediction confidence above the threshold
        predicted_intents = [self.classes[i] for i, p in enumerate(predictions) if p > threshold]
        return predicted_intents

    def generate_response(self, intent_tag):
        """
        Generate a response based on the provided intent tag.
        Args:
            intent_tag (str): Intent tag for which a response is to be generated.
        Returns:
            response (str): Randomly selected response for the provided intent tag.
        """
        # Iterate through intents in the intents data
        for intent in self.intents["intents"]:
            # Check if the current intent's tag matches the provided intent_tag
            if intent["tag"] == intent_tag:
                # Randomly select a response from the intent's responses list
                response = random.choice(intent["responses"])
                return response

    def train(self):
        """
        Train the chatbot using the provided intents data.
        Args:
            intents_data (dict): Intents data containing tags, patterns, and responses.
        Returns:
            None
        """
        # Load data from the provided intents_data and prepare
        self.prepare_data()
        # Create and train the deep learning model
        self.create_model()
        # Print a message to indicate that training is completed
        print("INFO: Training completed.")
        # Save the trained model
        model_dir = f'{BASE_DIR}/saved_models/chatbot_{self.language}'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model')
        self.model.save(model_path)
        print(f"INFO: Model saved at {model_path}")

    def gen_graph(self, hist, title):
        """
        Generate a graph for model accuracy and loss.

        Args:
            hist: The history object returned from model.fit().
            title (str): The title of the graph.
        """
        plt.plot(hist.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

        plt.plot(hist.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()