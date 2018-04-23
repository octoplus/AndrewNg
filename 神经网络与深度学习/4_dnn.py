# coding: utf-8
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

def sigmoid(x):
    return 1/(1+np.exp(-x)),x

def relu(x):
    return np.maximum(x,0),x

def relu_backward(dA,activation_cache):
    Z=activation_cache
    dZ=dA*np.where(Z>0,1,0)
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA,activation_cache):
    Z=activation_cache
    A=1/(1+np.exp(-Z))
    dZ = dA*A*(1-A)
    assert (dZ.shape == Z.shape)
    return dZ

def initialize_parameters_deep(layer_dims):
    """
    随机初始化参数
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])/100
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters

def linear_forward(A, W, b):
    """
    线性变换
    """
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    线性变换后进入激活函数
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    前向传播
    """
    caches = []
    A = X
    L = len(parameters) // 2 
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y):
    """
    计算损失函数
    """
    m = Y.shape[1]
    cost = -np.mean(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    """
    线性变换的反向传播
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.mean(dZ,1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    激活函数的反向传播
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    反向传播    
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -Y/AL+(1-Y)/(1-AL)
    
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    参数更新
    """
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate*grads["db" + str(l+1)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    最终模型
    """
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)
        
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    return parameters

def predict(parameters, X):
    AL, cache = L_model_forward(X, parameters)
    predictions = np.where(AL>0.5,1,0)
    return predictions

X,y= load_breast_cancer(True)

def standlize(x):
    """
    对数据进行max-min标准化
    """
    return (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))

X,y = standlize(X),standlize(y)

X,y=X.T,y.reshape(1,-1)

X_train,X_test,y_train,y_test=X[:,:400],X[:,400:],y[:,:400],y[:,400:]

layers_dims = [30, 5,5, 1]

parameters = L_layer_model(X_train, y_train, layers_dims,learning_rate=0.01, num_iterations = 300000, print_cost = True)

Y_prediction_train=predict(parameters,X_train)
Y_prediction_test=predict(parameters,X_test)
# 验证
# 两层 5
# 300000个迭代
# train accuracy: 100.0 %
# test accuracy: 94.0828402367 %

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))