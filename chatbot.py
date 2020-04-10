import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
import pickle

import json
with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w!= "?" ]
    words = sorted(list(set(words)))
    #print("words:", words)

    training = []
    output = []
    out_empty = [ 0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        #print("doc:", doc)
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])]=1

        training.append(bag)
        output.append(output_row)

    training=numpy.array(training)
    output=numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()
model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='softmax'))
#model.add(Activation('relu'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training, output, epochs=1000, batch_size=8)
#input= [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#input=numpy.array(input)
#print(input)
#print(input.shape)
#results = model.predict(input)
#print("results:", results)




def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
         
        print(bag_of_words(inp, words))
        results = model.predict([[bag_of_words(inp, words)]])
        print("Results:", results)
        results_index = numpy.argmax(results)
        print("Result_index:", results_index)
        tag = labels[results_index]
        print("tag:", tag)

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
