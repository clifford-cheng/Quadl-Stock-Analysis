#!/usr/bin/env python
# coding: utf-8

# In[205]:


import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# In[206]:


df = quandl.get('WIKI/GOOGL')
print(df.head())


# In[207]:


df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]


# In[208]:


df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100


# In[209]:


df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100


# In[210]:


df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]


# In[211]:


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True) 


# In[212]:


forecast_out = int(math.ceil(.01*len(df)))


# In[213]:


df['label'] = df[forecast_col].shift(-forecast_out)
df.tail()


# In[214]:


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


# In[215]:


df.dropna(inplace = True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)


# In[216]:


clf = LinearRegression()
clf.fit(X_train, y_train)


# In[217]:


accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[218]:


forecast_set = clf.predict(X_lately)


# In[219]:


print(forecast_set, accuracy, forecast_out)


# In[220]:


df['Forecast'] = np.nan
df.head()


# In[221]:


df.iloc[-1]


# In[222]:


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
print(df.head)


# In[223]:


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


# In[224]:


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:





# In[ ]:




