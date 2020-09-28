#!/usr/bin/python
# -*- coding:utf-8 -*-

import  numpy as np
import  matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 500

x1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num_observations)
x2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_observations)
# def multivariate_normal(mean, cov, size=None, check_valid=None, tol=None) 
# 生成多元正态分布函数

X = np.vstack((x1,x2)).astype(np.float32)
# x是1000*2维的数组
# x1全都是0，x2全都是1
Y = np.hstack((-np.ones(num_observations),np.ones(num_observations)))
# y是1000*1的数组，代表是标签

rate = 0.01
error = 0.05
size = 100

#----------------训练找到w和b----------------
def classify(train_data, train_label):
    # initialize weight = 0
    w = np.zeros(len(train_data[0]))
    bias = 0.0
    num = len(train_data)
    j = 0
    while True:
        j+=1
        for i in range(num):
            x = train_data[i]
            y = train_label[i]
            if y * (np.dot(w, x) + bias) <= 0:  # if prediction is wrong
                w = w + rate * y * x.T  # update weight:w = w + r * x_i * y_i
                bias = bias + rate * y  # update b:b = b + r * y_i

        loss = 0
        for i in range(num):
            y = train_label[i]
            output = np.dot(w, train_data[i]) + bias
            if output * y <= 0: 
                loss += 1
        print('This is ', j, 'th' )
        if (loss / num) <= error:
            print('w: ', w, 'bias: ', bias)
            return w, bias

    return w, bias

def show(X0, X1, W, b):
    plt.grid()
    plt.scatter(X0[:,1],X0[:,2],c = 'r',marker='o',s=500)
    plt.scatter(X1[:,1],X1[:,2],c = 'g',marker='*',s=500)
    p1=[-2.0,2.0]
    p2=[(W[0]+2*W[1])/W[2],(W[0]-2*W[1])/W[2]]
    plt.plot(p1,p2)
    plt.show()

if __name__ == '__main__':

    w, bias = classify(X, Y)
    show(X0, X1, w, bias)


