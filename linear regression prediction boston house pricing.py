#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from pandas_profiling import ProfileReport
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV , Lasso ,Ridge , RidgeCV , ElasticNet ,ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle


# In[2]:


boston = load_boston()


# In[3]:


df = pd.DataFrame(boston.data , columns=boston.feature_names)


# In[4]:


df['price'] = boston.target


# In[5]:


df.head()


# Lets check the dataset have any missing or na values. What are the relations between feature.etc
# with pandas profiling

# In[6]:


a = ProfileReport(df)
a.to_widgets()


#  creating a linear regression model

# In[8]:


LR = LinearRegression()


# In[9]:


y = df['price']
x= df.drop(columns=['price' ])


# In[10]:


LR.fit(x,y)


# In[11]:


LR.score(x,y)


# calculating r^2

# In[12]:


def adjr2(r):
    le = r.score(x,y)
    adjr = 1 - ((1-le)*(506-1))/(506-13-1)
    return(adjr)
adjr2(LR)


# In[ ]:


LR.intercept_


# In[146]:


LR.coef_


# In[153]:


LR.score(x,y)


# In[14]:


x_train , x_test,y_train,y_test=train_test_split(x,y , test_size= 0.10 , random_state=100)


# In[155]:


LR.fit(x_train,y_train)


# In[156]:


LR.score(x_test,y_test)


# The accuracy of the data has been increased after splitting it.

# In[157]:


LR.predict([[0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2.0,242.0,17.8,396.90,9.14]])


# Lets build a lasso model

# In[158]:


lassocv = LassoCV(alphas= None , cv = 50,max_iter= 1000,normalize= True)
lassocv.fit(x_train,y_train)


# In[159]:


lassocv.alpha_


# In[160]:


lsm = Lasso(alpha=lassocv.alpha_)
lsm.fit(x_train,y_train)


# In[161]:


lsm.score(x_test,y_test)


# In[162]:


lsm.predict([[0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2.0,242.0,17.8,396.90,9.14]])


# building Ridge model

# In[163]:


Rd = RidgeCV(cv = 10,normalize= True)
Rd.fit(x_train,y_train)


# In[164]:


Rd.alpha_


# In[165]:


ridge  = Ridge(alpha= Rd.alpha_)


# In[166]:


ridge.fit(x_train,y_train)


# In[167]:


ridge.score(x_test,y_test)


# In[168]:


ridge.predict([[0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2.0,242.0,17.8,396.90,9.14]])


# Building XGB model

# In[169]:


xb = XGBRegressor()


# In[170]:


xb.fit(x_train,y_train)


# In[171]:


xb.score(x_test,y_test)


# Building randomforest model

# In[15]:


rd = RandomForestRegressor()
rd.fit(x_train,y_train)


# In[16]:


rd.score(x_test,y_test)


# In[17]:


rd.predict([[0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2.0,242.0,17.8,396.90,9.14]])


# random forest has the best prediction till now
# 

# Lets create a elastic regressor model
# 

# In[175]:


elastic = ElasticNetCV(alphas= None , cv = 10)
elastic.fit(x_train,y_train)


# In[176]:


elastic.alpha_


# In[177]:


elastic.l1_ratio_


# In[178]:


elsaticlr = ElasticNet(alpha= elastic.alpha_ ,l1_ratio= elastic.l1_ratio_)


# In[179]:


elsaticlr.fit(x_train,y_train)


# In[180]:


elsaticlr.score(x_test,y_test)


# In[184]:


elsaticlr.predict([[0.02731,0.0,7.07,0.0,0.469,6.421,78.9,4.9671,2.0,242.0,17.8,396.90,9.14]])


# In[ ]:





# dumping the random forest model using pickle

# In[189]:


pickle.dump(rd,open('admi_pred_mo_rd.pickle' , 'wb'))


# In[18]:


ls


# In[ ]:




