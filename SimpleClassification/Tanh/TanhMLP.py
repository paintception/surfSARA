import numpy as np
import keras
import seaborn

from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2, activity_l2
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)

print "Reading the data"

X = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Positions.npy')
y = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Labels.npy')

X = preprocessing.scale(X)
dimension_input = X.shape[1]

new_y = np.asarray(y)

encoder = LabelEncoder()
encoder.fit(new_y)
encoded_y = encoder.transform(new_y)
dummy_y = np_utils.to_categorical(encoded_y)

print "Finished reading the data: ready for the MLP"

def plots(history):
	
	f1 = plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f1.savefig('/home/matthia/Desktop/MSc.-Thesis/Results/Figures/Tanh/AccuracyPlotModel1_'+str(learning_r)+'.png')
	plt.close()

	f2 = plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f2.savefig('/home/matthia/Desktop/MSc.-Thesis/Results/Figures/Tanh/LossPlotModel1_'+str(learning_r)+'.png')
	plt.close()

	print "Saving Results"

def baseline_model(sgd):

	model = Sequential()
	model.add(Dense(2048, input_dim=dimension_input, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(3, init='normal', activation='tanh'))
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model

def main():

	learning_rates = [0.01, 0.001, 0.0005, 0.0001, 0.00001]

	for learning_r in learning_rates:

		sgd = SGD(lr=learning_r, decay=1e-6, momentum=1, nesterov=True)

		print "Training for lr: ", learning_r

		model = baseline_model(sgd)
		X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.10, random_state=42)
		history = model.fit(X_train,y_train, nb_epoch=1000000, batch_size=5, verbose=1, validation_data=(X_test, y_test)) #callbacks=[tbCallBack])
	
		scores = model.evaluate(X,dummy_y)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

		model.save('KerasModels/Tanh/Model1_'+str(learning_r)+'.h5')
		plots(history)

if __name__ == '__main__':
	main()

