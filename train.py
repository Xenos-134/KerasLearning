#!/usr/bin/python
#==============================================================#
#                       TRAIN MODEL                            #                                        
#==============================================================#

from cProfile import label
from http.client import TEMPORARY_REDIRECT
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
from hands_data_extraction import * 
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


encoder = LabelBinarizer()


TEMPORARY_MODEL_NAME = "hands_model"

#==============================================================#
#                       TEST PART OF CNN                       #                                        
#==============================================================#
def test_train2():
    dataset = tf.data.TextLineDataset(['./out.txt'])
    dataset_labels = tf.data.TextLineDataset(["out_labels.txt"])

    train_dataset = []
    labels_dataset = []

    for element in dataset.as_numpy_iterator():
        #POR ENQUANTO VOU APENAS GERAR UMA LISTA UNIDIMENSIONAL
        text_array = element.decode().replace("[", "").replace("]", "").replace(",", "").split()
        numeric_array = [float(x) for x in text_array]
        train_dataset.append(numeric_array)

    for element in dataset_labels.as_numpy_iterator():
        labels_dataset.append(element.decode().replace(",", ""))

    labels_dataset = generate_labels(labels_dataset)
    #print(labels_dataset)

    #SPLIT TO TRAIN
    train_dataset, t_train_dataset, labels_dataset, t_labels_dataset = train_test_split(train_dataset, labels_dataset, test_size=0.1)


    
    print("==================== [ START ]==========================")
   
    model = models.Sequential()
    model.add(layers.Dense(32, input_shape=(63, ) ,activation='relu')) #Os neuronios 'e aleatorio
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(27, activation='softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()
    #print(labels_dataset)

    history = model.fit(train_dataset, labels_dataset, epochs=10, 
                        validation_data=(train_dataset, labels_dataset))
                        
    print("==================== [ FINISHED ]==========================")
    test_loss, test_acc = model.evaluate(train_dataset, labels_dataset, verbose=2)
    print("===========================================================")

    #save_model(model, TEMPORARY_MODEL_NAME)

    #MAKING PREDICTION TO GENERATE CONFUSION MATRIX
    predictions = model.predict(t_train_dataset)
    #rounded_predictions = np.argmax(predictions)
    rounded_predictions = []
    for elm in predictions:
        rounded_predictions.append(np.argmax(elm))

    print(len(rounded_predictions))
    print("\n\n\n")
    print(len(t_labels_dataset))
    cm = confusion_matrix(y_true=t_labels_dataset, y_pred=rounded_predictions)
    plot_confusion_matrix(cm, labels)


labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    df_cm = pd.DataFrame(cm, index = [i for i in labels],
                  columns = [i for i in labels])
    plt.figure(figsize = (15,15))
    #sn.heatmap(df_cm, annot=True)
    plt.show()


def generate_labels(train_labels):
    new_labels_list = []
    for label in train_labels:
        new_labels_list.append(labels.index(label))
    return new_labels_list
    

def save_model(model, model_name):
    print("Saved Model: ", model_name)
    model.save("./"+model_name)
    save_wights(model, model_name)


def save_wights(model, file_name):
    print("Saved Weights: ", file_name)
    model.save_weights('./'+file_name+'.h5')


def get_model(file_name):
    print("Loading model and weights: ", file_name)
    model = tf.keras.models.load_model("./"+file_name)
    model.load_weights('./'+file_name+'.h5')
    #model.summary()
    #print(model.get_weights())
    return model


def predict(image_name, model): #Only For single image for now
    print("PREDICTING LABEL FOR: ", image_name)
    out_data = str(read_one_image(image_name, False)).replace("[", "").replace("]", "").replace(",", "").split()
    #print(model.predict([train_dataset[-15]]))
    numeric_array = [float(x) for x in out_data]
    print("Predicted: ", labels[np.argmax(model.predict([numeric_array]))])
    print("Predicted: ", model.predict([numeric_array]))

def main():
    #flags = sys.argv
    #input_file = open(sys.argv, 'r')
    #input_file.close() 
    test_train2()

    #loaded_model = get_model(TEMPORARY_MODEL_NAME)
    #predict("test.jpg", loaded_model)


main()
