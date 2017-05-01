import numpy as np
import time

def read_data():

	X = []
	y = []
	General_X = []
	ind=1

	with open("/home/matthia/Desktop/MSc.-Thesis/Datasets/Merged.txt", 'r') as f:

		print "Reading the Data"					

		for line in f:
			record = line.split(";")
			pieces = [eval(x) for x in record[0:12]]
			piece = [item for sublist in pieces for item in sublist]
			piece = [item for sublist in piece for item in sublist]	

			X.append(piece)
			y.append(float(record[12][:-2]))

	X = np.asarray(X)
	
	print "Flipping the Data"

	odd_numbers = [k for a,k in enumerate(y) if a%2 != 0]
	even_numbers = [k for a,k in enumerate(y) if a%2 == 0]
	flipped_odd = [-a for a in odd_numbers]

	y = []

	for i,j in zip(even_numbers, flipped_odd):																																																															
		y.append(i)
		y.append(j)

	#if len(X) > len(y):
		#X = X[:-1].copy()

	new_y = []
	Pos_X = []							

	for pos, evaluation in zip(X,y):
		if evaluation >= - 1.5 and evaluation <= 1.5:
			Pos_X.append(pos)
			new_y.append("Equal")
	
		elif evaluation > 1.5 and evaluation <= 3.0:
			Pos_X.append(pos)
			new_y.append("WWB")

		elif evaluation > 3.0 and evaluation <= 5.0:
			Pos_X.append(pos)
			new_y.append("WW")

		elif evaluation > 5.0:
			Pos_X.append(pos)
			new_y.append("WCW")

		elif evaluation < 1.5 and evaluation >= -3.0:
			Pos_X.append(pos)
			new_y.append("BWB")

		elif evaluation < -3.0 and evaluation >= -5.0:
			Pos_X.append(pos)
			new_y.append("BW")

		elif evaluation < -5.0 and BCW:
			Pos_X.append(pos)
			new_y.append("BCW")
			
	new_y = np.asarray(new_y)

	np.save('~/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/CompleteEvaluation/ComplexPositions.npy', Pos_X)
	np.save('~/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/CompleteEvaluationComplexEvaluations.npy', new_y)

	print len(Pos_X)
	print len(new_y)

def main():
	read_data()

if __name__ == '__main__':
	main()
