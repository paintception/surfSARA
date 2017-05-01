import numpy as np
import argparse
#import cv2
#import seaborn
import keras
import os
import time 

from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder

def load_Positions():
    X = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/TripleEvaluation/600000Positions.npy')

    return X

def load_Labels():
    y = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/TripleEvaluation/600000Labels.npy')
    new_y = np.asarray(y)

    encoder = LabelEncoder()
    encoder.fit(new_y)
    encoded_y = encoder.transform(new_y)
    dummy_y = np_utils.to_categorical(encoded_y) 

    return dummy_y

def shape_data(dataset):
    
    dataset = np.reshape(dataset, (dataset.shape[0], 1, 8, 8))

    return dataset

def make_categorical(labels, n_classes):
    return(np_utils.to_categorical(labels, n_classes))

def MatFra_plots(history, i):

    print "Storing SabBido's Results for experiment: ", i

    f1 = plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./Desktop/MatFraAccuracyExp_'+str(i))

    #np.save('./res/MatFraTrainingAccuracyExp_'+str(i), np.asarray(history.history['acc']))
    #np.save('./res/MatFraValidationAccuracyExp_'+str(i), np.asarray(history.history['val_acc']))

    f2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./Desktop/MatFraLossExp_'+str(i))
    
    #np.save('./res/MatFraAccuracyLossExp_'+str(i), np.asarray(history.history['loss']))
    #np.save('./res/MatFraValidationLossExp_'+str(i), np.asarray(history.history['val_loss']))

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded")
    ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
    args = vars(ap.parse_args())

    X = load_Positions()
    X = shape_data(X)
    y = load_Labels()

    n_epochs = 2000
    opt = SGD(lr=0.01)
    cross_validation_exp = 1

    trainData, testData, trainLabels, testLabels = train_test_split(X, y, test_size=0.1, random_state=42)

    for i in xrange(0, cross_validation_exp):

        print "Running Experiment: ", i

        #trainData = load_Train_data()
        #trainLabels = load_Train_labels()
        #testData = load_Validation_data()
        #testLabels = load_Validation_labels()       

        tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/matthia/Desktop/logs', histogram_freq=0, write_graph=True, write_images=False)

        print("[INFO] compiling model...")

        model_MatFra = LeNet.build(width=8, height=8, depth=1, classes=3, mode=1, weightsPath=args["weights"] if args["load_model"] > 0 else None)
        model_MatFra.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        history_MatFra = model_MatFra.fit(trainData, trainLabels, batch_size=1028, nb_epoch=n_epochs, verbose=1, validation_data=(testData, testLabels), callbacks=[tbCallBack])

        print("[INFO] evaluating...")
        (loss, accuracy) = model_MatFra.evaluate(testData, testLabels, batch_size=1028, verbose=1)
        print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
        MatFra_plots(history_MatFra, i)

        #model_Google = LeNet.build(width=300, height=300, depth=3, classes=10, mode=3, weightsPath=args["weights"] if args["load_model"] > 0 else None)
        #model_Google.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        #history_Google = model_Google.fit(trainData, trainLabels, batch_size=50, nb_epoch=n_epochs, verbose=1, validation_data=(testData, testLabels))#,callbacks=[tbCallBack])
        
        #Google_plots(history_Google, i)

if __name__ == '__main__':
    main()
