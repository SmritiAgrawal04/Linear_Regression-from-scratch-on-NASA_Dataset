
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
get_ipython().magic('matplotlib inline')

from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score


# In[2]:


class Airfoil:
    dataset= []
    learning_rate= 0.8
    iterations= 1000
    test= []
    
    def predict(self, filename):
        self.test = pd.read_csv(filename, header= None)
        self.test= self.normalize(pd.DataFrame(self.test))
        
        X_test = np.c_[np.ones((len(self.test),1)),self.test]
        y_pred= X_test.dot(self.theta)
        return y_pred
    
    def  cal_cost(self,theta,X,y):
        length = len(y)

        predictions = X.dot(theta)
        cost = (1/2*length) * np.sum(np.square(predictions-y))
        return cost
    
    def modify_theta(self,length, theta, X_i, y_i, prediction):
        return theta -(1/length)*self.learning_rate*( X_i.T.dot((prediction - y_i)))
    
    def processing(self,length, X, y, theta):
        rand_ind = np.random.randint(0,length)
        X_i = X[rand_ind,:].reshape(1,X.shape[1])
        y_i = y[rand_ind].reshape(1,1)

        return X_i, y_i, np.dot(X_i,theta)
    
    def gradient_descent(self,X,y,theta):
        length = len(y)

        for it in range(self.iterations):
            cost =0.0
            for i in range(len(y)):
                X_i, y_i, prediction= self.processing(length, X, y, theta)
                theta= self.modify_theta(length, theta, X_i, y_i, prediction)
                cost += self.cal_cost(theta,X_i,y_i)

        return theta
    
    def LinearRegression(self, X, y):
        self.theta = np.random.randn(6,1)

        X_b = np.c_[np.zeros((len(X),1)),X]
        self.theta= self.gradient_descent(X_b,y,self.theta)
        
    
    def normalize(self,dfObj):
        minValuesObj = (dfObj.min()).tolist()

        maxValuesObj = (dfObj.max()).tolist()

        dfObj= dfObj.to_numpy()

        for i in range (dfObj.shape[1]):
            for j in range (dfObj.shape[0]):
                dfObj[j][i]= (dfObj[j][i]- (minValuesObj[i]))/(maxValuesObj[i]-minValuesObj[i])

        dfObj= pd.DataFrame(dfObj)
        return dfObj
    
    def train(self, filename):
        self.dataset = pd.read_csv(filename, header= None)
        labels= self.dataset.values[:,-1]
        self.dataset= np.delete(self.dataset.values, 5, axis=1)
        self.dataset= self.normalize(pd.DataFrame(self.dataset))
    
        train_data= self.dataset.to_numpy()        
        train_labels= labels

        self.LinearRegression(train_data, train_labels)

