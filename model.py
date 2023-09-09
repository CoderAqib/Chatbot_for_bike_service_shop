# Import necessary libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf

# Import NLTK library for natural language processing
import nltk
from nltk.stem import WordNetLemmatizer

# Create a WordNetLemmatizer object for text preprocessing
lemmatizer = WordNetLemmatizer()

# Load a JSON file containing questions and their associated tags
questions = json.loads(open('bike_service_queries.json').read())

# Initialize empty lists and variables for data processing
words = []         # List to store words in all queries
classes = []       # List to store question tags
documents = []     # List to store processed document-word pairs
ignoreLetters = ['?', '!', '.', ',']  # Characters to ignore

# Iterate through the questions and their queries in the JSON file
for question in questions['questions']:
    for query in question['queries']:
        # Tokenize each query into words
        wordList = nltk.word_tokenize(query)
        words.extend(wordList)  # Extend the words list with the tokenized words
        documents.append((wordList, question['tag']))  # Store the words and associated tag in documents list
        if question['tag'] not in classes:
            classes.append(question['tag'])  # Store unique question tags in the classes list

# Perform lemmatization on words and remove ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]

# Sort and make a unique set of words and classes
words = sorted(set(words))
classes = sorted(set(classes))

# Serialize and save the words and classes lists to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize variables and process documents for training data
training = []
outputEmpty = [0] * len(classes)

# Iterate through the processed documents
for document in documents:
    bag = []
    wordPatterns = document[0]
    # Lemmatize and convert words to lowercase in the document
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        # Create a bag of words (binary representation) for each document
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    # Create the output row representing the class of the document
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)  # Combine bag of words and output row

# Shuffle the training data to improve model training
random.shuffle(training)

# Convert the training data to a numpy array
training = np.array(training)

# Split the training data into input (X) and output (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Create a Sequential model using TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Define the stochastic gradient descent optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model with categorical crossentropy loss and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model on the training data for a specified number of epochs and batch size
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the trained model to a file
model.save('chatbot_model.h5', hist)

# Print a message to indicate that the code is done
print('Done')




