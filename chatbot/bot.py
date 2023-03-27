import json
import pickle
import numpy as np

from tensorflow.keras.models import load_model

from func import lemmatize_sentence, words_bag, predict_tag, get_response

# Load data
file = open(file='chatbot/files/intents.json')
intents = json.load(fp=file)
file.close()

file = open('chatbot/files/tags.pkl', 'rb')
tags = pickle.load(file=file)
file.close()

file = open('chatbot/files/words.pkl', 'rb')
words = pickle.load(file=file)
file.close()

# Load model
model = load_model('chatbot/model/chatbot_model.h5')

# Process
while True:
    message = input('You:')
    if message.lower() in ['e', 'exit', 'q', 'quit']:
        break

    else:
        intent = predict_tag(
            model=model,
            sentence=message,
            words=words,
            tags=tags
        )
        reply = get_response(intent, intents)
        print(f'Bot: {reply}')
