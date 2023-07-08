#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import pickle
import os
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn import metrics


# In[62]:


accxy = pd.read_csv('AccXYLatest.csv')
accxy


# In[63]:


X = accxy.drop(['Label'], axis=1)
Y = accxy.Label


# In[64]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[65]:


print(X_test)


# In[66]:


from sklearn.svm import SVC
model = SVC(C = 1, gamma =  1, kernel = 'rbf')


# In[67]:


model.fit(X_train, Y_train)


# In[68]:


model.score(X_test, Y_test)


# In[69]:


with open('svmfazarev.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[70]:


with open('svmfazarev.pkl', 'rb') as f:
    model = pickle.load(f)


# In[71]:


# buat meshgrid
h = 0.01  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Z berfungsi untuk melakukan predict terhadap model yang sudah dibuat
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #Ravel itu mengubah array nxn menjadi 1xn
Z = Z.reshape(xx.shape) #mengubah shape (pake fucntion ravel) Z jadi sama dengan shape xx

# Plot decision boundary dengan data poin
plt.contourf(xx, yy, Z, alpha=1) #plot contour
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, cmap=plt.cm.Paired) #plot scatter data kolom 1 dan 2, dengan tiap data memiliki wanra kontur yg sama

# Set batas plot dan label
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Accelerometer X')
plt.ylabel('Accelerometer Y')
plt.title("Plot Scatter Dengan Decision Boundary RBF")

# plot scatter grafik
plt.show()


# In[75]:


# from sklearn.model_selection import GridSearchCV

# #cari parameter terbaik
# parameter_grid = {'C': [0.1, 1, 10, 100, 1000],
#                   'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#                   'kernel': ['rbf']
#                  }

# grid = GridSearchCV(SVC(), parameter_grid, refit = True, verbose = 3)

# grid.fit(X_train, Y_train)


# In[76]:


# print(grid.best_params_)
# print(grid.best_estimator_)


# In[77]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data0 = data[data.Label==0]
data1 = data[data.Label==1]
plt.xlabel('Accelerometer X')
plt.ylabel('Accelerometer Y')
plt.scatter(data0['Accelerometer X'], data0['Accelerometer Y'], color = 'blue', marker = 'x')
plt.scatter(data1['Accelerometer X'], data1['Accelerometer Y'], color = 'red', marker = '.')
plt.title('Grafik yang dihasilkan')


# In[78]:


import pickle

# Load the trained model from the pickle file
with open('svmfazarev.pkl', 'rb') as file:
    model = pickle.load(file)

# Use the model for prediction or other tasks
result = model.predict(data)
print(result)

