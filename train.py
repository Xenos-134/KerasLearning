#!/usr/bin/python
#==============================================================#
#                       TRAIN MODEL                            #                                        
#==============================================================#

from cProfile import label
from statistics import mode
import sys
from tabnanny import verbose

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()


def train(input_file) :
    print("==================( CNN Training (START) )==================")
    #TODO
    print("==================( CNN Training   (END) )==================")



#==============================================================#
#                       TEST PART OF CNN                       #                                        
#==============================================================#
def test_train2():
    dataset = tf.data.TextLineDataset(['./out.txt'])
    labels = tf.data.TextLineDataset(['./labels.txt'])

    train_dataset = []
    labels_dataset = []

    for element in dataset.as_numpy_iterator():
        #POR ENQUANTO VOU APENAS GERAR UMA LISTA UNIDIMENSIONAL
        text_array = element.decode().replace("[", "").replace("]", "").replace(",", "").split()
        numeric_array = [float(x) for x in text_array]
        train_dataset.append(numeric_array)

    for element in labels.as_numpy_iterator():
        labels_dataset.append(element.decode().replace(",", ""))

    labels_dataset = generate_labels(labels_dataset)
    print(labels_dataset)

    #print(train_dataset)
   
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(63,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    #model.summary()

    history = model.fit(train_dataset, labels_dataset, epochs=10, 
                        validation_data=(train_dataset, labels_dataset))

    print("==================== [ FINISHED ]==========================")
    test_loss, test_acc = model.evaluate(train_dataset, labels_dataset, verbose=2)
    print("===========================================================")

    #save_model(model, "test_model")

labels = ["A", "B", "C", "D", "E", "F"]


def generate_labels(train_labels):
    new_labels_list = []
    for label in train_labels:
        new_labels_list.append(labels.index(label))

    return new_labels_list
    


def save_model(model, model_name):
    #TODO
    print("TEST")
    model.save("./"+model_name)
    save_wights(model, model_name)


def save_wights(model, file_name):
    model.save_weights('./'+file_name+'.h5')

def load_model(file_name):
    print("TEST")
    model = tf.keras.models.load("./"+file_name)


def load_weights(model, file_name):
    model.load_wights(file_name+'.h5')


def main():
    #flags = sys.argv
    #input_file = open(sys.argv, 'r')
    #input_file.close() 
    test_train2()

main()
