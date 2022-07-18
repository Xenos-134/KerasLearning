#!/usr/bin/python
#==============================================================#
#                       TRAIN MODEL                            #                                        
#==============================================================#
from cProfile import label
from http.client import TEMPORARY_REDIRECT
from statistics import mode
import sys
from tabnanny import verbose
from traceback import print_tb

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

from enum import  Enum
from tensorflow.python.client import device_lib

encoder = LabelBinarizer()


TEMPORARY_MODEL_NAME = "hands_model"
class TRAIN_METHODS(Enum):
    MEDIA_PIPE = 1,
    VGG16 = 2,

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


#==============================================================#
#              TRAIN USING MEDIA PIPE EXTRACTED DATA           #                                        
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
    print("Predicted Label: ", labels[np.argmax(model.predict([numeric_array]))])
    print("ALL PREDICTONS: ", model.predict([numeric_array]))


def get_flags(flags):
    flags_obj = {
        "file_path": "./test.jpg",
        "is_train": False, #DEFAULT IS TO TEST SOME IMAGE
        "algorithm": TRAIN_METHODS.MEDIA_PIPE, #DEFAULT DATA EXTRACTION ALGOs
    }

    if("-d" in flags): flags_obj['file_path'] = flags[flags.index("-d")+1]
    if("-t" in flags): flags_obj['is_train'] = True
    if("-a" in flags): 
        if("vgg16" in flags):
            flags_obj['algorithm'] = TRAIN_METHODS.VGG16
        else:
            flags_obj['algorithm'] = TRAIN_METHODS.MEDIA_PIPE

    print(flags_obj)
    return flags_obj

#==============================================================#
#              TRAIN USING MEDIA PIPE EXTRACTED DATA           #                                        
#==============================================================#
def train_vgg16(): #TODO DIR PATH AS ARG
    print("START TRAIN WITH VGG16 ALGO")
    #FOR NOW WILL USE EXTERNAL DATASET (CIFAR!))

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=2, 
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)


def vgg16_image_preprocess():
    print(">>>>>>>>>>> START TRAIN")

    custom_image_dataset = tf.keras.preprocessing.image_dataset_from_directory("./asl_train", labels='inferred');

    for elm in custom_image_dataset.as_numpy_iterator():
        print(elm)
        return

if __name__ == "__main__":
    flags = get_flags(sys.argv)
    #print(device_lib.list_local_devices())
    
    #train_vgg16()
    vgg16_image_preprocess()


''' 
    if(not flags['is_train']):
        loaded_model = get_model(TEMPORARY_MODEL_NAME)
        predict(flags['file_path'], loaded_model)

    else:
        if(flags['algorithm'] == TRAIN_METHODS.MEDIA_PIPE):
            input_file = open(sys.argv, 'r')
            input_file.close() 
            test_train2()
        elif(flags['algorithm'] == TRAIN_METHODS.VGG16):
            train_vgg16()
'''

