import numpy as np
import argparse
import cv2
import seaborn
import keras
import glob

from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from PIL import Image

def load_dataset():
	
	print "Loading the Data"
	
	filelist = glob.glob('/home/matthia/Desktop/MSc.-Thesis/CNNImages/BasicEvaluation/TotalImages/*.png')
	data = np.array([np.array(Image.open(fname)) for fname in filelist])

	data = data[:, np.newaxis, :, :]

	print "Data Loaded"

	return data

def load_labels():

	print "Loading Labels"

	labels = np.load('/home/matthia/Desktop/MSc.-Thesis/CNNImages/BasicEvaluation/TotalImages/Labels.npy')
	print len(labels)

	print "Labels Loaded"

	return labels

def shape_data(dataset):
	
	data = dataset.data.reshape((dataset.data.shape[0], 8, 8))
	shaped_data = data[:, np.newaxis, :, :]

	print shaped_data.shape

	return shaped_data

def make_categorical(labels, n_classes):
	return(np_utils.to_categorical(labels, n_classes))

def plots(history, learning_r):
	
	f1 = plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f1.savefig('/home/matthia/Desktop/Accuracy'+str(learning_r)+'.png')
	plt.close()

	f2 = plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f2.savefig('/home/matthia/Desktop/Loss_'+str(learning_r)+'.png')
	plt.close()

	print "Saving Results"

def final_prediction_test(model, testData, testLabels):	
	
	for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
		# classify the digit
		probs = model.predict(testData[np.newaxis, i])
		prediction = probs.argmax(axis=1)

		# resize the image from a 28 x 28 image to a 96 x 96 image so we
		# can better see it
		
		"""
		image = (testData[i][0] * 255).astype("uint8")
		image = cv2.merge([image] * 3)
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
		cv2.putText(image, str(prediction[0]), (5, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		# show the image and prediction
		"""
		print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
			np.argmax(testLabels[i])))
		"""
		cv2.imshow("Digit", image)
		cv2.waitKey(0)
		"""

def main():

	learning_rates = [0.5, 0.01, 0.001, 0.0005, 0.0001, 0.00001]

	for learning_r in learning_rates:

		print "Learning Rate: ", learning_r

		ap = argparse.ArgumentParser()
		ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
		ap.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded")
		ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
		args = vars(ap.parse_args())

		X = load_dataset()
		print(X.shape)
		#dataset = shape_data(data)
		y = load_labels()
		new_y = np.asarray(y)
		encoder = LabelEncoder()
		encoder.fit(new_y)
		encoded_y = encoder.transform(new_y)
		dummy_y = np_utils.to_categorical(encoded_y)
		
		trainData, testData, trainLabels, testLabels = train_test_split(X, dummy_y, test_size=0.10, random_state=42)

		tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/matthia/Desktop/ogs', histogram_freq=0, write_graph=True, write_images=False)

		print("[INFO] compiling model...")
		opt = SGD(lr=learning_r)

		model = LeNet.build(width=64, height=64, depth=1, classes=3, weightsPath=args["weights"] if args["load_model"] > 0 else None)
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

		history = model.fit(trainData, trainLabels, batch_size=128, nb_epoch=2, verbose=1, validation_data=(testData,testLabels),callbacks=[tbCallBack])

		print("[INFO] evaluating...")
		(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
		print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

		plots(history, learning_r)
		final_prediction_test(model, testData, testLabels)

if __name__ == '__main__':
	main()


