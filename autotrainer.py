import mnist_loader
import pickle
import numpy as np 
import autoencoder

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# only get x inputs
training_data, test_data = np.array(list(training_data))[:,0], np.array(list(test_data))[:,0]
training_data, test_data = list(map(lambda x: np.squeeze(x), training_data)), list(map(lambda x: np.squeeze(x), test_data))
print("processed data")

# [ 784, 400, L50, 400, 784]
autoencoder = autoencoder.Autoencoder([784, 400], [400, 784], 50)
print("finished building autoencoder")

autoencoder.train(training_data, test_data)
print("finished training!")


'''=
# Creates filename and prepares network properties(performance and parameters) for saving
# name follows naming scheme: inputsize_layer1size_layer2size_...outputsize 
file_name = str(net.sizes).replace("[","").replace("]","").replace(" ","").replace(",","_")+'.pkl'
new_net_properties = [net.performance, net.weights, net.biases, ]

Saves network properties (learned parameters and performance). 
If the shape had been previously trained, it will override previous saved file if 
new performance is better. If it is a new shape, it will save to a new file


try:
	with open('best_networks/{0}'.format(file_name), 'rb') as f:
		old_net_properties = pickle.load(f)
	if new_net_properties[0] > old_net_properties[0]:
		with open('best_networks/{0}'.format(file_name), 'wb') as f:
			pickle.dump(new_net_properties, f)
		print('Found a better version of network with shape {0}!'.format(file_name[:-4]))
	else:
		print('New network not better than previous')

	print(old_net_properties[0], "old")
	print(new_net_properties[0], "new")

except FileNotFoundError:
	print('New Network Shape!')
	with open('best_networks/{0}'.format(file_name), 'wb') as f:
		pickle.dump(new_net_properties, f)
	print("new performance", new_net_properties[0])

# if no test_data is given, performance cannot be compared
except TypeError:
	print('Must supply test_data to update file')'''