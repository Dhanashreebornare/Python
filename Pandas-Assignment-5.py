#!/usr/bin/env python
# coding: utf-8

# # Pandas Assignment-5

# In[1]:


#for loop to calculate number of customers getting techsupport and are male senior citizens
import pandas as pd


# In[3]:


df=pd.read_csv('D:\Intellipaat\Python\PANDAS\customer_churn.csv')
df


# In[7]:


num_customers=0
for index, row in df.iterrows():
    if row['TechSupport'] == 'Yes' and row['gender'] == 'Male' and row['SeniorCitizen'] == 1:
        num_customers += 1

# Print the number of customers meeting the criteria
print("Number of customers with techsupport who are male senior citizens:", num_customers)


# In[ ]:




