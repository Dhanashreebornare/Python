#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Pandas Assignment-1


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('D:\Intellipaat\Python\PANDAS\customer_churn.csv')
df


# In[3]:


newCols= df.iloc[: , [3, 7, 9, 20]].copy() 
newCols


# In[4]:


df.iloc[20:200,2:15]


# In[5]:


df.head(100)


# In[6]:


df.tail(1)


# In[7]:


df.sort_values( by=["tenure"], ascending=False)


# In[8]:


new_df1 = df[(df['tenure'] > 50) & (df['gender'] == 'Female')]

new_df1


# In[9]:


new_df2=df[(df['gender']=='Male') & (df['SeniorCitizen']==0)]
new_df2


# In[10]:


new_df3=df[(df['TechSupport']=='Yes') & (df['Churn']=='No')]
new_df3


# In[11]:


new_df4=df[(df['Contract']=='Month-to-month') & (df['Churn']=='No')]
new_df4


# In[15]:


count=0
for index, row in df.iterrows():
    if row['gender'] == 'Male' and row['TechSupport'] and row['SeniorCitizen']:
        count += 1

print("Number of male senior citizens getting tech support:", count)


# In[ ]:




