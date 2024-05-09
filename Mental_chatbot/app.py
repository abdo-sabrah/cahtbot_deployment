#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, jsonify,render_template,url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle


# In[ ]:



# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('mental_chatbot_model.keras')

# Load the words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the intents file
with open('mentalhealth.json', 'r') as json_data:
    intents = json.load(json_data)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Function to predict class/intent
def predict_class(sentence, model):
    ERROR_THRESHOLD = 0.25
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get response
def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "I'm sorry, I didn't understand that."

# List of greetings and exit commands
greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
exit_commands = ["quit", "pause", "exit", "goodbye", "bye", "later", "stop"]

# Route for chatbot response
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json['message']
    
    if user_input.lower() in greetings:
        return jsonify({'response': "Hello! How can I assist you today?"})
    if user_input.lower() in exit_commands:
        return jsonify({'response': "Goodbye! Have a great day."})
    
    ints = predict_class(user_input, model)
    res = getResponse(ints, intents)
    return jsonify({'response': res})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


# In[ ]:





# In[ ]:




