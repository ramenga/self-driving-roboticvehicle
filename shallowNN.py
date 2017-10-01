import numpy as np

'''
Attempt at a self contained simple self driving toy car
Will use minimum dependencies
Whole deep learning framework is coded in numpy here itself
end-to-end NN system, no hardcoded rules
Attempt to use GPGPU speedup in future
Remember to normalize inputs

NN with single hidden layer
You may want to run training on a PC and inference on the RPi
Andrew Ng's code is used as a reference for the NN
'''
def sigmoid(val):
	#sigmoid activation function
	sig = (1/(1+np.exp(-z)))
	return sig

def layer_sizes(X,Y):
	# returns layer sizes from X and Y data sizes
	# hidden layer size needs to be hardcoded
	n_x = X.shape[0] #input layer size
	n_h = 32         #hidden layer size
	n_y = Y.shape[0] #output layer size
	return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
	#initialize randomly the weights and biases
	W1 = np.random.randn(n_h,n_x)*0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h)
	b2 = np.zeros((n_y,1))
	parameters = {"w1":W1,"b1":b1,"W2":W2,"b2":b2}
	return parameters

def forward_propagation(X,parameters):
	#retrieve parameters from dict
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	
	#forward prop
	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1) #activation function
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2) #activation fn of output layer
	cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
	return A2,cache

def compute_cost(A2,Y,parameters):
	#cost for each epoch
	m = Y.shape[1] #no of training examples
	logprobs = (np.multiply(np.log(A2),Y))+(np.multiply(np.log(1-A2),1-Y))
	cost = (-1/m)*np.sum(logprobs)
	cost = np.squeeze(cost)
	assert(isinstance(cost,float))
	return cost

def backprop(parameters,cache,X,Y):
	#backpropagation
	#Calculation Of Gradients
	m = X.shape[1] #no of training examples
	#retrieve parameters
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	#retrieve activation outputs
	A1 = cache["A1"]
	A2 = cache["A2"]
	#calc gradients
	dZ2 = A2 - Y
	dW2 = (1/m)*np.dot(dZ2,A1.T)
	db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
	dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
	dW1 = (1/m)*np.dot(dZ1,X.T)
	db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
	#store grads in dict
	gradients = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}
	return gradients
	
def update_parameters(parameters,gradients,learning_rate=0.01):
	#Gradient Descent for one epoch
	#retrieve parameters
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	#retrieve gradients
	dW1 = gradients["dW1"]
	db1 = gradients["db1"]
	dW2 = gradients["dW2"]
	db2 = gradients["db2"]
	#update_parameters
	W1 = W1 - learning_rate*dW1
	b1 = b1 - learning_rate*db1
	W2 = W2 - learning_rate*dW2
	b2 = b2 - learning_rate*db2
	#store parameters after updating
	parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
	#return updated parameters
	return parameters

def model_nn(X,Y,n_h,num_iterations = 10000,print_cost = True)
	#Main Model for training the NN
	#retrieve layer sizes, the fn returns an array
	n_x = layer_sizes(X,Y)[0] #n[0]
	n_y = layer_sizes(X,Y)[2] #n[2]
	#random initialization of parameters
	parameters = initialize_parameters(n_x,n_h,n_y)
	#retrieve initialized parameters
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	#Gradient descent for num_iterations epochs
	for i in range(0,num_iterations):
		#forward propagation
		A2,cache = forward_propagation(X,parameters)
		#cost function
		cost = compute_cost(A2,Y,parameters)
		#backpropagation
		gradients = backprop(parameters,cache,X,Y)
		#update parameters
		parameters = update_parameters(parameters,gradients)
		#print cost every 1000 iterations
		if print_cost and i%1000==0:
			print("Cost after iteration %i:%f"%(i,cost))
	return parameters


