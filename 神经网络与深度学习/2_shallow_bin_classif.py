# coding: utf-8
# 单个感知机
# 二分类任务
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# 准备数据
# 按照AndrewNg的教程，数据排列方式为 (n,m)
# n为特征数量，m为样本数量

X,y= load_breast_cancer(True)

def standlize(x):
    """
    对数据进行max-min标准化
    """
    return (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))

X,y = standlize(X),standlize(y)

X,y=X.T,y.reshape(1,-1)


X_train,X_test,y_train,y_test=X[:,:400],X[:,400:],y[:,:400],y[:,400:]

# print X_train.shape
# print y_train.shape

# 搭建感知机

def sigmoid(z):
    """
    激活函数
    """
    return 1/(1+np.exp(-z))

def initialize_with_zeros(dim):
    """
    初始化参数，赋值为较小的随机数
    """
    w=np.random.randn(dim,1)/100
    b=np.random.randn()/100
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
	# 样本数量
    m = X.shape[1]
    
    # 前向传播
    A = sigmoid(np.dot(w.T,X)+b) #计算激活值

    cost=-np.mean(Y*np.log(A)+(1-Y)*np.log(1-A)) #计算损失函数
    
    # 反向传播
    dw = np.dot(X,(A-Y).T)/m
    db = np.mean(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        
        # 计算损失和梯度
        grads, cost = propagate(w, b, X, Y)
        
        # 获取梯度
        dw = grads["dw"]
        db = grads["db"]
        
        # 梯度下降
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
    预测函数
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        Y_prediction[0,i]= 1 if A[0,i]>0.5 else 0
    
    assert(Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

# 训练&预测
# train accuracy: 95.75 %
# test accuracy: 96.449704142 %

d = model(X_train, y_train, X_test, y_test, num_iterations = 3000, learning_rate = 0.05, print_cost = True)

