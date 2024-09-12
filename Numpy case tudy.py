#!/usr/bin/env python
# coding: utf-8

# # Numpy- Case study

# 1. Create a function that takes dimensions as tuples e.g. (3, 3) and a numeric
# value and returns a NumPy array of the given dimension filled with the
# given value e.g.: solve((3, 3), 5) will return
# [
# [5, 5, 5],
# [5, 5, 5],
# [5, 5, 5]
# ]
# 
# 

# In[1]:


import numpy as np
arr=np.full((3,3),5)
print(arr)


# 2. Create a method that takes n NumPy arrays of the same dimensions,
# sums them and returns the answer.

# In[8]:


import numpy as np
n= int(input('Enter the number of arrays-'))
R = int(input("Enter the number of rows:"))
C= int(input("Enter the number of columns:"))
for i in range(0,n):
    
    print("Enter the entries in a single line (separated by space): ")
# User input of entries in a 
# single line separated by space
    entries = list(map(int, input().split()))
 
# For printing the matrix
    matrix = np.array(entries).reshape(R, C)
    print('Matrix=',matrix)
i==i+1
# adding n arrays
for i in range(0,n):
    ans=np.sum([matrix])
    ans=ans+np.sum([matrix])
i==i+1
print('Sum of n matrices =',ans)



# 3. Given a 2 D Array of N X M Dimension, write a function that accepts this
# array as well as two numbers N and M. The method should return the
# top-left N X M sub matrix, e.g:
# [
# [1, 2, 3],
# [4, 5, 6],
# [7, 8, 9],
# ]
# top_left_sub_matrix (matrix, 2, 2) -> should return:
# [
# [1, 2]
# [4, 5]
# ]
# 
# 

# In[9]:


import numpy as np
R = int(input("Enter the number of rows:"))
C= int(input("Enter the number of columns:"))
 
 
print("Enter the entries in a single line (separated by space): ")
 
# User input of entries in a 
# single line separated by space
entries = list(map(int, input().split()))
 
# For printing the matrix
matrix = np.array(entries).reshape(R, C)
print(matrix)
N=int(input('Enter number of rows to be extracted-'))
M=int(input('Enter number of columns to be extracted-'))
res = matrix[0:N:1,0:M:1]
# Display result
print("Created Submatrix:\n",res,"\n")



# 4. Given a 2 D Array of N X M Dimension, write a function that accepts this
# array as well as two numbers N and M. The method should return the
# bottom-right N X M sub matrix, e.g:
# [
# [1, 2, 3],
# [4, 5, 6],
# [7, 8, 9],
# ]
# sub_matrix(matrix, 1, 1) -> should return : (Keep in mind these arrays are
# zero indexed)
# [
# [5, 6]
# [8, 9]
# ]
# 
# 

# In[11]:


import numpy as np
R = int(input("Enter the number of rows:"))
C= int(input("Enter the number of columns:"))
 
 
print("Enter the entries in a single line (separated by space): ")
 
# User input of entries in a 
# single line separated by space
entries = list(map(int, input().split()))
 
# For printing the matrix
matrix = np.array(entries).reshape(R, C)
print(matrix)
N=int(input('Enter number of rows to be extracted-'))
M=int(input('Enter number of columns to be extracted-'))
res = matrix[0-N::1,0-M::1]
# Display result
print("Created Submatrix:\n",res,"\n")


# 5. Given a 1 D NumPy Array. Write a function that accepts this array as
# parameters. The method should return a dictionary with 'mean' and
# 'std_dev' as key and array's mean and array's standard deviation as
# values:
# [1, 1, 1]
# solution(arr) -> should return :
# {'mean': 1.0, 'std_dev': 0.0}

# In[13]:


import numpy as np
print('Enter the values of an array-')
entries = list(map(int, input().split()))
arr=np.array(entries)
print('array=',arr)
print("mean= % a "  %(np.average(arr)),"std deviation= % s "  %(np.std(arr)))


# In[ ]:




