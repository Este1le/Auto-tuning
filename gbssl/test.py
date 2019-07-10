import numpy as np

n = 5
d = 3
x = np.random.rand(n,d)
y = np.random.rand(n,1)
w = np.random.rand(n,n)

def total():
    total = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(n):
            total += (y[i] - y[j]) * w[i][j] * (x[i]-x[j])
    return total

def sub1():
    total = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(n):
            total += y[i] * w[i][j] * x[i]
    return total

def sub2():
    total = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(n):
            total += y[i] * w[i][j] * x[j]
    return total

def sub3():
    total = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(n):
            total += y[j]*w[i][j]*x[i]
    return total

def sub4():
    total = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(n):
            total +=y[j] * w[i][j]*x[j]
    return total

def msub3():
    return np.dot(np.dot(x.T,w),y)


def loss():
    total = 0
    for i in range(n):
        for j in range(n):
            total += np.square(y[i] - y[j]) * w[i][j]
    total /= 2
    return total

def mloss():
    D = np.diag(np.sum(w, axis=1))
    print(D)
    print(w)
    return np.dot(np.dot(y.T, D-w),y)




