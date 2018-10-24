import mnist_loader
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# only get x inputs
training_data = np.array(list(training_data))[:,0]

training_data = np.array(list(map(lambda x: np.squeeze(x), training_data)))
print(training_data[0:2])
print(np.shape(training_data))
