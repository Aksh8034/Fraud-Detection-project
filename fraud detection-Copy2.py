#!/usr/bin/env python
# coding: utf-8

# In[3]:


#imported all the necesarry libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


#dataset to a panda dataframe
data=pd.read_csv('Fraud (1).csv')


# In[5]:


#to read data
data.head() 


# In[6]:


data.tail()


# In[7]:


#data information
data.info()


# In[8]:


#data cleaning
data.isnull().sum()


# In[9]:


#distribution of legit transaction and fraudlent transaction
data['isFraud'].value_counts()


# In[10]:


#this dataset is imbalance
#separating data for analysis
legit=data[data.isFraud==0]
fraud=data[data.isFraud==1]


# In[11]:


print(legit.shape)
print(fraud.shape)


# In[12]:


#statistical measures
legit.amount.describe()


# In[13]:


fraud.amount.describe()


# In[14]:


#compare the values for both transactions
data.groupby('isFraud').mean()


# In[15]:


#using under sampling - dataset sample with similar distribution
legit_sample=legit.sample(n=1142)


# In[16]:


#concatination
#axis=0, rows
#axis=1, cols
new_data=pd.concat([legit_sample,fraud],axis=0)
new_data.head()


# In[17]:


new_data.tail()


# In[18]:


new_data['isFraud'].value_counts()


# In[19]:


new_data.groupby('isFraud').mean() #confirm that the nature of data is same, helps in knowing the good dataset

#The variables were selected in accordance with the model requirements and analysis process. The most relevant variables were amount, old balnaceorig, new balanceorig, old balancedest, new balancedest as the values were able to put forth the realation and help in fraud detection.
#Therefore rest of the variables were dropped. 

#The key factors in fraud detection are the balance history (new,old), the destination account's balance history(new, old) and the overall amount history. The miscalculation and evident imbalance helps in fraud detection. Also, the numbers are the key to detect fraud, therefore a high focus should be played on these factors.However, in some cases the type and time may also play a key role where complexity is involoved. 
# In[20]:


X=new_data.drop(['nameDest','type','nameOrig','step','isFlaggedFraud'],axis=1)
print(X)


# In[21]:


#two variable formed for further ML model
M=X.drop(columns='isFraud',axis=1)
Y=new_data['isFraud']
print(M)


# In[22]:


print(Y)


# In[23]:


#split the data into training data and testing data
M_train, M_test, Y_train, Y_test=train_test_split(M,Y,test_size=0.2, stratify=Y, random_state=2)


# In[24]:


print(M.shape, M_train.shape,M_test.shape)


# model training 
# using logistics regression model since it is used genrally used for binary classification. 

# In[25]:


model=LogisticRegression()


# In[26]:


#training logistic regression model
model.fit(M_train,Y_train)


# In[27]:


#model evaluation on basis of accuracy score
#accuracy on training data 
M_train_prediction=model.predict(M_train)
training_data_accuracy=accuracy_score(M_train_prediction,Y_train)


# In[28]:


print("Accuracy on training data: ",training_data_accuracy)


# In[29]:


#accuracy in accordane with test data
M_test_prediction=model.predict(M_test)
test_data_accuracy=accuracy_score(M_test_prediction,Y_test)


# In[30]:


print("Accuracy on test data: ",test_data_accuracy)


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


sns.pairplot(new_data, hue='isFraud')


# In[32]:


#testing other models 
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(M_train, Y_train)


# In[33]:


print('Decision Tree Training accuracy', tree.score(M_train,Y_train)) #overfitting in decision tree


# In[34]:


from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(M_train,Y_train)


# In[35]:


print('random forest Training accuracy', forest.score(M_test,Y_test)) #over fitting 

In conclusion, we understood the accuracy difference of the models. However, we choose and worked with Logistic regression because it is the best model for binary classification and also, to overcome the challenge of over fitting which is visible in desicion tree and random forest. Since the world is moving ahead with empowered technology that is advanced everyday,therefore, to avoid the frauds they should update their systems using a better versions of hardwares and anti-malware softwares.

However, talking from a data scientist point of view, an AI powered ML model which is built with all the probablistics scenerios of fraud keeping in mind every complex outcome should be built and used by the company. 

If the model or software is placed in the company the next step is to maintain and kepp updating it with time to avoid the new technics of fraud and keeping it running in all due situations. A team of proficient data scinetist, analyst and engineers should be set in place for maximum output from the software. 