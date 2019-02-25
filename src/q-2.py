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
