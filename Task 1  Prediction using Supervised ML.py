#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation Internship In Data Science & Buisness Analytics

# # Name : Vishakha Jamkhedkar
# 

# task name : prediction using supervised ML
# ~ Predict the percentage of an student based on the no. of study hours.
# ~ This is a simple linear regression task as it involves just 2 variables.
# 

# In[1]:


# importing necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading dataset

data = pd.read_csv("http://bit.ly/w-data")
data.head()


# In[3]:


#getting size/dimensions of data set
data.shape


# In[4]:


#getting measures of central tendency
data.describe()


# In[5]:


# plotting the data

data.plot(x = 'Hours',y = 'Scores', style = 'o')
plt.title('Hours Vs. Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores obtained')
plt.show()


# In[6]:


#dividing data into independent and dependent variables


# In[7]:


X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[8]:


#import sklearn library


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


# In[11]:


#training dataset


# In[12]:


from sklearn.linear_model import LinearRegression

#Creating an object of linear regression
reg = LinearRegression()


# In[13]:


#fitting the model
reg.fit(X_train, y_train)


# In[14]:


l = reg.coef_*X+reg.intercept_

plt.scatter(X,y)
plt.plot(X,l)
plt.show()


# In[15]:


#to retreive the intercept and coeffient

print("intercept:", reg.intercept_)
print("coeffient:", reg.coef_)


# In[16]:


y_pred = reg.predict(X_test)


# In[17]:


y_pred


# In[18]:


#comparing Actual vs Predicted values

df = pd.DataFrame({'Actual':y_test, 'predicted': y_pred})
df


# In[19]:


#visualizing actual and predicted values

plt.scatter(X_test,y_test)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("test data actual values")
plt.show()

plt.scatter(X_test,y_pred, marker = 'v')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("test data predicted values")
plt.show()


# In[20]:


#Evavulating the algorithm with absoulte error, mean squared error, root mean squared error

from sklearn import metrics
print("Mean absoulte error:",metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error:",metrics.mean_squared_error(y_test, y_pred))
print("Root Mean squared error:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[21]:


#What will be predicted score if a student studies for 9.25 hrs/ day?

l = reg.coef_*9.25+reg.intercept_
l


# # What will be predicted score if a student studies for 9.25 hrs/ day?
# 

# # by applying linear regression the answer will be, if student studies for 9.25 hrs/day so the student will score 92.91 marks.

# In[ ]:




