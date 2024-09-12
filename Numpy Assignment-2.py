#!/usr/bin/env python
# coding: utf-8

# # Numpy Assignment-2

# In[1]:


#1. Create a 3x3 matrix array with values ranging from 2 to 10.
import numpy as np
x=np.arange(2,11).reshape(3,3)
print(x)


# In[2]:


#2. Create a NumPy array having user input values and convert the integer type to the float type of the elements of the array.
#For instance: Original array [1, 2, 3, 4] Array converted to a float type: [ 1. 2. 3. 4.]
y=np.array([1,2,3,4])
y=np.asfarray(y)
print(y)


# In[3]:


#3. Write a NumPy program to append values to the end of an array. 
#For instance: Original array: [10, 20, 30] . After that, append values to the end of the array: [10 20 30 40 50 60 70 80 90]
arr=np.array([10,20,30])
a=np.append(arr,[10,20,30,40,50,60,70,80,90])
print(a)


# In[4]:


#4. Create two NumPy arrays and add the elements of both the arrays and store the result in sumArray.
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
sumArray=arr1+arr2
print(sumArray)


# In[14]:


#5. Create a 3x3 array having values from 10-90 (interval of 10) and store that in array1
arr=np.arange(10,91,10).reshape(3,3)
print(arr)
#Perform the following tasks:
#a. Extract the 1st row from the array
print('\n Extracting 1st row from the array', arr[0])
#b. Extract the last element from the array
print('\n Extracting last element from the array', arr[2,[2]])


# In[ ]:





# In[ ]:




