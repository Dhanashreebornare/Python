#!/usr/bin/env python
# coding: utf-8

# # Pandas Assignment-4

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv('D:\Intellipaat\Python\PANDAS\customer_churn.csv')
df


# In[8]:


newdf=df.sort_values(by='tenure', ascending=False)
newdf


# In[17]:


#fetching  records where 'tenure'>50 and gender='Female'
df1=df[(df['tenure']>50) & (df['gender']=='Female')]
df1


# In[24]:


#b. gender as 'male' and 'senior citizen as '0'
df2=df[(df['gender']=='Male') & (df['SeniorCitizen']==0)]
df2


# In[26]:


#c. TechSupport as 'yes' and 'churn' as 'no'
df3=df[(df['TechSupport']=='Yes') & (df['Churn']=='No')]
df3


# In[29]:


# Contract type as'month to month' and churn as 'yes'
df4=df[(df['Contract']=='Month-to-month') & (df['Churn']=='Yes')]
df4


# In[ ]:




