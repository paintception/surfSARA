import numpy
import pandas
import seaborn
import numpy as np
import time
 
from matplotlib import pyplot as plt 
from sklearn import preprocessing
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2, activity_l2
from sklearn.model_selection import train_test_split
from keras.utils.visualize_util import plot

seed = 7
numpy.random.seed(seed)

X = []
y = []

with open("", 'r') as f:
	for line in f:
		record = line.split(";")
		pieces = [eval(x) for x in record[0:12]]
		piece = [item for sublist in pieces for item in sublist]
		piece = [item for sublist in piece for item in sublist]	
		X.append(piece)
		y.append(float(record[12][:-2]))

X = np.asarray(X)
y = np.asarray(y)

X = preprocessing.scale(X)

dimof_input = X.shape[1]

def plots(history):
	
	f1 = plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f1.savefig('/home/matthia/Desktop/MSc.-Thesis/Results/Figures/Regression/Tanh/AccuracyPlotModel1_'+str(learning_r)+'.png')
	plt.close()

	f2 = plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f2.savefig('/home/matthia/Desktop/MSc.-Thesis/Results/Figures/Regression/Tanh/LossPlotModel1_'+str(learning_r)+'.png')
	plt.close()

	print "Saving Results"

def run_model(sgd):
	# create model
	model = Sequential()
	model.add(Dense(1,input_dim=dimof_input, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(1, init='normal', activation='linear'))
	model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
	
	return model

def main():

    learning_rates = [0.01, 0.001, 0.0005, 0.0001, 0.00001]

	for learning_r in learning_rates:

	    sgd = SGD(lr=learning_r, decay=1e-6, momentum=1, nesterov=True)
	    print "Training for lr: ", learning_r

            model = run_model(sgd)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
            history = model.fit(X_train,y_train, nb_epoch=300, batch_size=50, verbose=1, validation_data=(X_test, y_test))
            scores = model.evaluate(X,y)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            model.save('KerasModels/Regression/Tanh/Model1_'+str(learning_r)+'.h5')
            plots(history)
	
if __name__ == '__main__':
	main()
	
