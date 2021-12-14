
#import our library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Get our dataset
train_data = np.genfromtxt('iris-Train.data', delimiter=',', names=True)
test_data = np.genfromtxt('iris-Test.data', delimiter=',', names=True)


#Define our features and target 
x1 = np.transpose(np.asmatrix(train_data['x1']))
x2 = np.transpose(np.asmatrix(train_data['x2']))
x = np.append(x1, x2, 1)
y = np.transpose(np.asmatrix(train_data['y']))

#-----------------------------------------------------------

xtest1 = np.transpose(np.asmatrix(test_data['x1']))
xtest2 = np.transpose(np.asmatrix(test_data['x2']))
xtest = np.append(xtest1, xtest2, 1)
ytest = np.transpose(np.asmatrix(test_data['y']))


#Define theta and x_0
ones = np.ones((x.shape[0],1))
onestest = np.ones((xtest.shape[0],1))
one_x = np.append(ones, x, 1)
one_xtest = np.append(onestest, xtest, 1)

m, n = np.shape(one_x)
theta = np.zeros((1,n))
theta = np.transpose(theta)
theta_new = theta


#main function for binary classification(logistic Regression)

#Define our iteration & Learning Rate 
loops= 1000
alpha = 0.002
#--------------------------------

#Define our MSE function with Gradient
mse = np.zeros((loops,2))
r = 0

while r < loops:    
    
    h_theta = 1/(1+np.exp(-1*(one_x*theta)))
    gradient = one_x.T*(h_theta-y)
    theta = theta - alpha*gradient    
    y_pred = 1/(1+np.exp(-1*(one_x*theta)))
    
    loop_mse = 0
    for i in range(0, m):       
        
        loop_mse += -(y.A1[i]*np.log(1/(1+np.exp(-1*(one_x[i]*theta))))) - ((1-y.A1[i])*np.log(1-(1/(1+np.exp(-1*(one_x[i]*theta))))))
    loop_mse = loop_mse/(m)
    mse[r,0] = r
    mse[r,1] = loop_mse
    
    r += 1
    
print('Number of iteration = ',loops)
print('Learning Rate = ',alpha)
print('Theta[0] = ', theta.A1[0])
print('Theta[1] = ', theta.A1[1])
print('Theta[2] = ', theta.A1[2])


#Calculate our h(theta)
y_pred = 1/(1+np.exp(-1*(one_x*theta)))
y_pred_test = 1/(1+np.exp(-1*(one_xtest*theta)))


#calculate train MSE
train_mse = 0
m = y.shape[0]
for i in range(m):
    train_mse += ((y.A1[i] - y_pred.A1[i])**2)
train_mse = train_mse/(2*m)
print("Train MSE = ", train_mse)


#calculate test MSE
test_mse = 0
m = ytest.shape[0]
for i in range(m):
    test_mse += ((ytest.A1[i] - y_pred_test.A1[i])**2)
test_mse = test_mse/(2*m)
print("Test MSE = ", test_mse)


#classification our dataset in 2 class
minx1 = np.min(x1)
maxx1 = np.max(x1)
x2_1 = -(theta.A1[0] + theta.A1[1]*minx1)/theta.A1[2]
x2_2 = -(theta.A1[0] + theta.A1[1]*maxx1)/theta.A1[2]



#Plot our dataset and MSE
plt.scatter(train_data['x1'],train_data['x2'], c=train_data['y'], s = 10, label = 'Train data')
plt.scatter(train_data['x1'],train_data['x2'], c=train_data['y'], s = 10, label = 'Train data')
plt.legend(loc = 'upper left')
plt.plot((minx1,maxx1), (x2_1,x2_2))
plt.show() 

plt.scatter(test_data['x1'],test_data['x2'], c=test_data['y'], s = 10, label = 'Test data')
plt.scatter(test_data['x1'],test_data['x2'], c=test_data['y'], s = 10, label = 'Test data')
plt.legend(loc = 'upper left')
plt.plot((minx1,maxx1), (x2_1,x2_2))
plt.show() 

plt.plot(mse.T[0], mse.T[1], color = "black", label = 'MSE') 
plt.legend(loc = 'upper right')
plt.show() 