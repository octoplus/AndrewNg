# coding: utf-8
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.datasets import load_breast_cancer
import sklearn.linear_model

X,y= load_breast_cancer(True)

def standlize(x):
    """
    对数据进行max-min标准化
    """
    return (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))

X,y = standlize(X),standlize(y)

X,y=X.T,y.reshape(1,-1)

X_train,X_test,y_train,y_test=X[:,:400],X[:,400:],y[:,:400],y[:,400:]

def sigmoid(z):
    """
    激活函数
    """
    return 1/(1+np.exp(-z))

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h,n_x)/100
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)/100
    b2 = np.zeros((n_y,1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1] 

    logprobs = Y*np.log(A2)+(1-Y)*np.log(1-A2)
    cost = -np.mean(logprobs)
    
    cost = np.squeeze(cost)

    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.mean(dZ2,1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-A1**2)
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.mean(dZ1,1,keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    
    A2, cache = forward_propagation(X, parameters)
    predictions = np.where(A2>0.5,1,0)
    
    return predictions

parameters = nn_model(X_train, y_train, n_h = 4, num_iterations = 30000, print_cost=True)

Y_prediction_train=predict(parameters,X_train)
Y_prediction_test=predict(parameters,X_test)

# 验证
# train accuracy: 100.0 %
# test accuracy: 97.0414201183 %

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
