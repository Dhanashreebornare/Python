#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("D:\Intellipaat\Python\Python online assignments\CS-Go-Project\csgo_round_snapshots.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[ ]:


df.columns


# In[58]:


df.info()


# In[59]:


df.describe()


# In[60]:


# There are no null values present in the dataset
df.isnull().sum().sum()


# In[61]:


df["map"].value_counts()


# In[62]:


# lets see teams how they are successful in winning rounds
counts = df['map'].value_counts()
total = counts.sum()
percentages = counts / total * 100

for map_name, count, percent in zip(counts.index, counts.values, percentages.values):
    print(f'{map_name}: {percent:.2f}%','/',count)


# In[63]:


plt.bar(counts.index, counts.values)

plt.xticks(rotation=45)
plt.xlabel('Map')

plt.ylabel('Count')


# In[64]:


for i in df.columns:
  if (df[i].dtypes=="object") | (df[i].dtypes=="bool"):
    print("Columns which have categorical values",i)


# In[65]:


df["bomb_planted"].value_counts()


# In[66]:


df["map"].value_counts()


# In[67]:


df["round_winner"].value_counts()


# In[68]:


# Converting categorical features into a integer column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["bomb_planted"]=le.fit_transform(df["bomb_planted"])
df["map"]=le.fit_transform(df["map"])
df["round_winner"]=le.fit_transform(df["round_winner"])


# In[69]:


X=df.drop(columns=["round_winner"])
y=df[["round_winner"]]


# In[70]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[71]:


# Scaling the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[72]:


# Applying Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[73]:


lda=LinearDiscriminantAnalysis()


# In[74]:


lda.fit(x_train,y_train)


# In[75]:


lda.transform(x_test)


# In[76]:


# Obtaining the LDA coefficients.This will give the importance scores associated with each feature.


# In[77]:


lda_coefficients=np.exp(np.abs(lda.coef_))


# In[78]:


#to collapse multiple dimension to one dimensions
lda_coefficients= lda_coefficients.flatten()


# In[79]:


lda_coefficients


# In[80]:


num_features=X.shape[1]


# In[81]:


num_features


# In[82]:


feature_indices=np.arange(num_features)


# In[83]:


feature_indices


# In[84]:


feature_names=list(X.columns)


# In[85]:


plt.figure(figsize=(20,18))
plt.bar(feature_indices,lda_coefficients)
plt.xticks(feature_indices,feature_names,rotation="vertical")
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance Scores')
plt.show()


# In[86]:


# Performing feature selection using LDA by using the absolute values of the
# LDA coefficients as a measure of feature importance.


# In[87]:


df_feature_score=pd.DataFrame({"Feature_names":feature_names,"feature_scores":lda_coefficients})


# In[88]:


top_20_values=df_feature_score.nlargest(20,'feature_scores')


# In[89]:


# Selecting the top 20 features based on the feature importance


# In[90]:


top_20_values.head(20)


# In[91]:


top_20_values.index


# In[92]:


x_train=x_train[:,[17, 8, 65, 7, 40, 5, 6, 20, 15, 4, 89, 21, 64, 18, 9, 16, 87, 14,
            10, 12]]


# In[93]:


x_test=x_test[:,[17, 8, 65, 7, 40, 5, 6, 20, 15, 4, 89, 21, 64, 18, 9, 16, 87, 14,
            10, 12]]


# In[94]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[95]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[96]:


from sklearn.metrics import classification_report


# In[97]:


print(classification_report(y_test,y_pred))


# In[98]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)


# In[99]:


accuracy_score(y_test,y_pred)


# In[100]:


print(classification_report(y_test,y_pred))


# In[101]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)


# In[102]:


accuracy_score(y_test,y_pred)


# In[103]:


print(classification_report(y_test,y_pred))


# In[104]:


# Support Vector Machine


# In[105]:


from sklearn.svm import SVC
svc=SVC()


# In[ ]:


svc.fit(x_train,y_train)


# In[ ]:


y_pred=svc.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# KNN


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knc= KNeighborsClassifier()
knc.fit(x_train,y_train)
y_pred=knc.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# XGBoost
import xgboost as xgb
xgbc= xgb.XGBClassifier()


# In[ ]:


xgbc.fit(x_train,y_train)


# In[ ]:


y_pred=xgbc.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# Clearly we can see that random forest is the best model for this dataset.

