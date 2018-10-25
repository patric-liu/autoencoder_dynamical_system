import numpy as np

def array_2_vec(img_array):
	shape = np.shape(img_array)
	vector_size = shape[0] * shape[1]
	img_vec = np.zeros( (vector_size,1) )
	for i in range(shape[1]):
		for j in range(shape[0]):
			value = img_array[i, j]
			index = i * shape[1] +j
			img_vec[index] = value
	return np.array(np.squeeze(img_vec))

def vec_2_array(img_vec, shape):
	img_array = np.zeros( shape )
	for i in range(shape[1]):
		for j in range(shape[0]):
			index = i * shape[1] + j
			value = img_vec[index]
			img_array[i, j] = value
			
	return img_array