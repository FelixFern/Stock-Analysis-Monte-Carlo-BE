import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

# Read files
file = open(file='intents.json')
intents = json.load(fp=file)
file.close()

# Storing lists
words = []
tags = []
documents = []
ignores = [
    '[', ']', '@', '-', '_', '!', '#', '$', 
    '%', '^', '&', '*', '(', ')', '<', '>', 
    '?', '/', '|', '}', '{', '~', ':'
]

# Loop for every subkeys, subvalues in intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Lemmatize the words and rid of any duplicates
cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in ignores]
cleaned_words = sorted(set(cleaned_words))

# Save temporary cleaned words and tags
files = [open('words.pkl', 'wb'), open('tags.pkl', 'wb')]
pickle.dump(obj=cleaned_words, file=files[0])
pickle.dump(obj=tags, file=files[1])

for file in files:
    file.close()

# Preparing dataset
training = []
for document in documents:
    bag = []
    word_pattern, tag = document[0], document[1]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]

    for word in cleaned_words:
        if word in word_patterns:
            bag.append(1)

        else:
            bag.append(0)

    idx = tags.index(tag)
    output = [0 for _ in range(len(tags))]
    output[idx] = 1
    training.append([bag, output])

# Shuffle the dataset
random.shuffle(training)

# Save the training data
file = open('training.pkl', 'wb')
pickle.dump(obj=training, file=file)
file.close()