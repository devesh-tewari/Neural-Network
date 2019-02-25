#!/usr/bin/env python
# coding: utf-8

# # Question 1
# ## Train and validate your own n-layer Neural Network
# 
# #### Implementation details:
# 1. The neural network is modeled in a class called 'Neural_Network' It has all the functions required to train and test the model which will be described later.
# 2. I have used cross entropy as a measure of cost because it a classification problem and cross entropy suits well for such problems.
# 3. The neural network implemented is flexible enough to have as many hidden layes as we want and any number of nodes at each layer.
# 4. The activation functions available are sigmoid, softmax, relu and tanh.
# 5. Best results are obtained with only one hidden layer with size 128, learning rate 0.005, number of epochs 9, hidden layer activation function sigmoid, output layer activation function softmax and batch size of 100 samples. 

# In[3]:


import pandas as pd
import numpy as np
import copy
import math
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import os

csv_path = 'Apparel/apparel-trainval.csv'#raw_input("Enter path to input CSV file: ")
dataset = pd.read_csv(csv_path)

dataset = dataset.astype('float64')

dataset.head()


# We first normalize all the data in the following code

# In[4]:


Attributes = dataset.keys()[1:]
Label = dataset.keys()[0]

mean = {}
std = {}
for x in Attributes:
    mean[x] = np.mean(dataset[x])
    std[x] = np.std(dataset[x])
    
normalized_dataset = dataset
for x in Attributes:
    normalized_dataset[x] = (normalized_dataset[x] - mean[x]) / std[x]
    
#split data into train data and validation data
splitted = np.split(normalized_dataset, [int(.8 * len(dataset))])
train_data = splitted[0]
validation_data = splitted[1]
    
X = train_data[Attributes].values
assert (len(X[X != X]) == 0) # No nan should be present

Y = train_data[Label].values


# The following function one hot encodes all the labels

# In[5]:


def one_hot_encode(labels):
    num_labels = len(labels)
    unique = len(np.unique(labels))
    one_hot_encode = np.zeros((num_labels, unique))
    one_hot_encode[np.arange(num_labels), labels] = 1
    return one_hot_encode


# The class Neural_Netwok has the following functions:
# 1. **\__init\__**: Initializes the parameters of the network, i.e. initializes input layer size, output layer size, number of hidden layers, size of each hidden layer, hidden layers' activation function, output layer activation function. Also it initializes the weights and bias.
# 2. **forward**: It propagates the inputs through the netwok and return the output from the last layer.
# 3. **backward**: Calculates the gradient of cross entropy with respect to weights and bias and updates the weights and bias according to the gradient descent formula.
# 4. **fit**: Used to train the neural network. For each epoch it iterates the data forward and backward batch by batch.
# 5. **predict**: Predicts the class labels of data. It actually returns the probabilities of each sample belonging to all the classes.
# 6. **evaluate**: Given the predictions and actual labels, it prints the accuracy and plots loss with respect to iterations during training.

# In[ ]:


class Neural_Network:
    def __init__(self, s1, s2, s3, s4_list, hidden_act, output_act, n):
        # Define Hyperparameters
        self.inputLayerSize = s1
        self.outputLayerSize = s2
        self.hiddenLayerCount = s3
        self.hiddenLayerSizes = s4_list  # s4_list = [hiddenLayer1Size, hiddenLayer2Size, ...]
        self.hiddenActivation = hidden_act  # 'sigmoid', 'relu', 'tanh' or 'softmax'
        self.outputActivation = output_act  # 'sigmoid', 'relu', 'tanh' or 'softmax'
        self.learningRate = n
        
        # Weights (parameters)
        self.W = []
        self.W.append( np.random.rand(self.inputLayerSize, self.hiddenLayerSizes[0]) )
        for i in range(self.hiddenLayerCount-1):
            self.W.append( np.random.rand(self.hiddenLayerSizes[i], self.hiddenLayerSizes[i+1]) )
        self.W.append( np.random.rand(self.hiddenLayerSizes[self.hiddenLayerCount-1], self.outputLayerSize) )
    
        for i in range(len(self.W)):
            self.W[i] = self.W[i] * math.sqrt(2.0/self.W[i].shape[0])
            
        # Bias
        self.b = [None]*(self.hiddenLayerCount+1)
        for i in range(len(self.b)):
            self.b[i] = np.random.rand( 1,self.W[i].shape[1] )
                
        
    def forward(self, x):
        # Propagate inputs through network
        self.z = []  # z2, z3, ... z(n+1) n=hiddenLayerCount
        self.a = []  # a2, a3, ... a(n)
        
        self.z.append( np.dot(x, self.W[0]) + self.b[0] )
        
        for i in range(self.hiddenLayerCount):
            cur_a = self.activations[self.hiddenActivation]( self.z[i] )
            self.a.append( cur_a )
            self.z.append( np.dot(cur_a, self.W[i+1]) + self.b[i+1] )
            
        yHat = self.activations[self.outputActivation]( self.z[self.hiddenLayerCount] )
        
        return yHat
            
            
    def backward(self, x, y, yHat):
        dJdW = [None] * len(self.W)
        dJdb = [None] * len(self.b)
        
        a_idx = len(self.a)-1
        z_idx = len(self.z)-1
    
        if self.outputLayerSize == 1:
            y = y.reshape(len(y),1)
        
        ####### For output weights #######
        delta = (yHat-y)
        dJdW[ len(dJdW)-1 ] = np.dot( self.a[a_idx].T, delta )
        
        dJdb[ len(dJdb)-1 ] = delta
        ##################################
        
        a_idx -= 1
        z_idx -= 1
        
        ####### For hidden weights #######
        for i in range(len(dJdW)-2, 0, -1):
            
            df_z = self.derivatives['d_'+self.hiddenActivation]( self.z[z_idx] )
            
            delta = np.multiply( np.dot(delta, self.W[i+1].T), df_z )
            
            dJdW[i] = np.dot( self.a[a_idx].T, delta )
            dJdb[i] = delta
            
            a_idx -= 1
            z_idx -= 1
        ##################################
        
        ####### For input weights ########
        df_z = self.derivatives['d_'+self.hiddenActivation]( self.z[z_idx] )
        delta = np.multiply( np.dot(delta, self.W[1].T), df_z )
        dJdW[0] = np.dot( x.T, delta )
        dJdb[0] = delta
        ##################################
        
        # Update all the weights
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.learningRate * dJdW[i]
            self.b[i] = self.b[i] - self.learningRate * dJdb[i]
            
            
    def fit(self, x, y, num_ephocs=5, batch_size=2000):
        self.loss = []
        
        self.batch_size = batch_size
        
        for i in range(num_ephocs):
            
            start = 0
            end = batch_size
            
            while end <= x.shape[0]:
                
                # Extract a batch
                x_batch = x[start:end,:]
                y_batch = y[start:end,:]
                
                # Iterate the batch over the network
                yHat = self.forward(x_batch)
                self.backward(x_batch, y_batch, yHat)
                
                start += batch_size
                end += batch_size
                
                loss = np.sum(-y_batch * np.log(yHat))
                self.loss.append( loss )

    
    def predict(self, x):
        start = 0
        end = self.batch_size
        yHat = np.array([])
        while end <= x.shape[0]:
            x_batch = x[start:end,:]
            yHat_batch = self.forward(x_batch)
            if len(yHat) == 0:
                yHat = yHat_batch
            else:
                yHat = np.vstack((yHat,yHat_batch))
            start += self.batch_size
            end += self.batch_size
        return yHat
    
    
    def evaluate(self, yHat, y):
        T, F = 0, 0
        for i in range(y.shape[0]):
            if np.argmax(y[i]) == np.argmax(yHat[i]):
                T += 1
            else:
                F += 1
                
        print "Validation Result:"
        print "Accuracy = ", float(T)/(T+F)
        
        plt.figure(figsize=(15,10))
        plt.plot(range(len(self.loss)), self.loss, label='loss')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend()
        plt.title('loss w.r.t. number of iterations')
        plt.show()
        
    
    # Activation functions
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))
    
    def relu(z):
        ret = np.maximum(0,z)
        return ret
    
    def tanh(z):
        return np.tanh(z)
    
    def softmax(z):
        e_z = np.exp(z - z.max(1).reshape(z.shape[0],1))
        ret = e_z / np.sum(e_z,1).reshape(z.shape[0],1)
        return ret
    
    
    # Derivatives of activation functions
    def d_sigmoid(z):
        sigmoid_z = 1.0 / (1 + np.exp(-z))
        return sigmoid_z*(1-sigmoid_z)
    
    def d_relu(z):
        ret = copy.deepcopy(z)
        ret[ret <= 0] = 0
        ret[ret > 0] = 1
        return ret
    
    def d_tanh(z):
        tanh_z = np.tanh(z)
        return 1 - (tanh_z)**2
    
    def d_softmax(z):
        e_z = np.exp(z - z.max(1).reshape(z.shape[0],1))
        sm = e_z / np.sum(e_z,1).reshape(z.shape[0],1)
        ret = sm*(1-sm)
        return ret
    
    # Set of activation functions available
    activations ={'sigmoid': sigmoid,
                 'relu': relu,
                 'tanh': tanh,
                 'softmax': softmax
                 }
    
    # set of derivatives of activation functions
    derivatives ={'d_sigmoid': d_sigmoid,
                 'd_relu': d_relu,
                 'd_tanh': d_tanh,
                 'd_softmax': d_softmax
                 }


# In[10]:


# One hot encode the labels
encoder = LabelEncoder()
Y = train_data[Label].values
encoder.fit(Y)
Y = encoder.transform(Y)
Y = one_hot_encode(Y)

Y_validation = validation_data[Label].values
encoder.fit(Y_validation)
Y_validation = encoder.transform(Y_validation)
Y_validation = one_hot_encode(Y_validation)


# #### Now we actully create an instance of Neural_Network and train and validate the data
# #### The parameters at which highest accuracy is achieved are shown

# In[15]:


inputLayerSize = 784
outputLayerSize = 10
hiddenLayerCount = 1
hiddenLayerSizes = [128]
hiddenLayerActivation = 'sigmoid'
outputLayerActivation = 'softmax'
learningRate = 0.005

NN = Neural_Network( inputLayerSize, outputLayerSize, hiddenLayerCount, hiddenLayerSizes,
                     hiddenLayerActivation, outputLayerActivation, learningRate )

NN.fit(X, Y, num_ephocs=9, batch_size=100)

predictions = NN.predict(validation_data[Attributes].values)

NN.evaluate(predictions, Y_validation)


# #### Observations:
# 1. We see that loss reduces overall as the number of iterations increase that implies correctness of our implementation.
# 2. The accuracy on the validation is high, i.e. **87.66%**

# The following code reads the test data and saves it in the required file.

# In[96]:


csv_path = 'Apparel/apparel-test.csv'
test_dataset = pd.read_csv(csv_path)

test_dataset = test_dataset.astype('float64')

mean = {}
std = {}
for x in Attributes:
    mean[x] = np.mean(test_dataset[x])
    std[x] = np.std(test_dataset[x])
    
normalized_dataset = test_dataset
for x in Attributes:
    normalized_dataset[x] = (normalized_dataset[x] - mean[x]) / std[x]

X_test = np.array(normalized_dataset)

predictions = NN.predict(X_test)

with open('output data/2018201039_prediction.csv', 'w') as f:
    for i in range(len(predictions)):
        f.write( str(np.argmax(predictions[i])) + os.linesep )


# ### Plotting prediction loss as a function of number of hidden layers

# In[76]:


loss_arr = []

for i in range(1,11):
    NN = Neural_Network(784,10,i,[12]*i,'sigmoid','softmax', 0.005)

    NN.fit(X, Y, num_ephocs=2, batch_size=100)

    predictions = NN.predict(validation_data[Attributes].values)

    loss = np.sum(-Y_validation * np.log(predictions))
    loss_arr.append(loss)

plt.figure(figsize=(15,10))
plt.plot(range(1,len(loss_arr)+1), loss_arr, label='prediction loss')
plt.xlabel('number of hidden layers')
plt.ylabel('prediction loss')
plt.legend()
plt.title('prediction loss w.r.t. number of hidden layers')
plt.show()


# #### Observations:
# 1. Increasing the number of hidden layers might improve the accuracy or might not, it really depends on the complexity of the problem that you are trying to solve.
# 2. In our case it increases, the reasons for this increase might be because of overfitting or the training data is not enough to have multiple hidden layers.

# ### Plotting error as a function of number of epochs

# In[16]:


error_arr = []

for i in range(1,21):
    NN = Neural_Network(784,10,1,[128],'sigmoid','softmax', 0.005)

    NN.fit(X, Y, num_ephocs=i, batch_size=100)

    predictions = NN.predict(validation_data[Attributes].values)
    
    T, F = 0, 0
    for i in range(predictions.shape[0]):
        if np.argmax(Y_validation[i]) == np.argmax(predictions[i]):
            T += 1
        else:
            F += 1
    error_arr.append( float(F)/(T+F) )

plt.figure(figsize=(15,10))
plt.plot(range(1,len(error_arr)+1), error_arr, label='error')
plt.xlabel('number of epochs')
plt.ylabel('error')
plt.legend()
plt.title('error w.r.t. number of epochs')
plt.show()


# #### Observations:
# 1. As the number of epochs increase, the error reduces.
# 2. The reason is that our gradient descent algorithm reaches more close to the minima as we increase number of epochs.
# 3. But after some point it starts to increase due to overfitting.

# # Question 2
# ## House Price Prediction
# ###### In this question, we are required to report on how we would modify our above neural network for such task with proper reasoning.

# The neural network designed in question 1 is designed for predicting class labels which is a classification problem. But this problem requires us to predict the house price which is a regression problem.
# 
# In our implementation we took the loss function as cross entropy which is a good fit for classification problems, but in the house prediction problem, we need some other loss function which may be mean square error or mean absolute error or mean percentage error. Choosing mean square error for this task is a good choice.
# 
# Changing the loss function from cross entropy to mean square error will cause change in the calculation of gradients. Currently we are calculation gradients of cross entropy with respect to weights and bias in the backward function, they need to be updated for mean square error as cost.
# 
# Since in regression problems, we have a single output, we need to change the number of nodes in output layer to one that can be done by simply passing the outputLayerSize parameter as one when we initialize our neural network.
# 
# In the house prediction problem, it would be a good choice to use relu activation function for the hidden layers and a linear activation function for the output layer since it is a resression problem and output can be any positive real number. In our implementation, we will need to add another activation function 'linear'.
# We can also use relu at the output layer since the house price will always be greater than zero.
# 
# The number of hidden layers will be more than one for this problem as the output will be a highly non linear function of the inputs, the exact number of hidden layers and number of nodes in each hidden layer can only be found by experimenting the validation results.
# 
# In question 1, we only had numerical attributes of pixel values that range from 0 to 255. But for the problem of house price prediction, there are several attributes that are not numerical but categorical. A good way to account for categorical features is to one hot encode them. This will result in increase in number of attributes but is a reliable way to handle categorical attributes. In our implementation, this can be done by reading the numerical and categorical attributes separately, then one hot encoding the categorical attributes and then finally combining all the attributes to form the train and validation data.
# 
# Another way to account for categorical values is to use one input for each category and scale the integer values, e.g. (0,...,11) for month to continuous values in the range of other input variables. However, this approach would assume that you have some hierarchy in the categories, let's say February is 'better' or 'higher' than January.
# 
# The optimal number of epochs and batch size can be decided by graph plotting as we did in question one. We will need to choose these parameters that result in minimum loss.

# In[ ]:




