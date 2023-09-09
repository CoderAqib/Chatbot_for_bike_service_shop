# Import necessary libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Create a WordNetLemmatizer object for text preprocessing
lemmatizer = WordNetLemmatizer()

# Load JSON data from 'bike_service_queries.json'
questions = json.load(open('bike_service_queries.json', 'r'))

# Load preprocessed data (words, classes, and a pre-trained chatbot model)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Define a function to clean up a sentence by tokenizing and lemmatizing it
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Define a function to convert a sentence into a bag of words (binary representation)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Define a function to predict the class of a sentence and its probability
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'query': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Define a function to get a response based on the predicted class
def get_response(questions_list, questions_json):
    tag = questions_list[0]['query']
    list_of_questions = questions_json['questions']
    for i in list_of_questions:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Main loop for the chatbot
print("Bot is running! Ask anything.")

while True:
    message = input("")
    if message == 'exit':
        print(exit())
    else:
        ints = predict_class(message)
        res = get_response(ints, questions)
        print(res, "\n -------------------------------------------------------------------------------")
  