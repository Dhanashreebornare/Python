#!/usr/bin/env python
# coding: utf-8

# # Data structure assignment-1
# 

# 1. Create a list named ‘myList’ that has the following elements: 10, 20, 30,
# ‘apple’, True, 8.10:
# a. Now in the ‘myList’, append these values: 30, 40
# b. After that, reverse the elements of the ‘myList’ and store that in
# ‘reversedList’
# 

# In[1]:


mylist=[10,20,10,'apple', True, 8.10]
print(mylist)


# In[2]:


mylist.extend([30,40])
mylist


# 2. Create a dictionary with key values as 1, 2, 3 and the values as ‘data’,
# ‘information’ and ‘text’:
# a. After that, eliminate the ‘text’ value from the dictionary
# b. Add ‘features’ in the dictionary
# c. Fetch the ‘data’ element from the dictionary and display it in the output
# 

# In[16]:


Dict={1:'data',2:'information', 3:'text'}
print('Dictionary=', Dict)


# In[17]:


#a. After that, eliminate the ‘text’ value from the dictionary
del(Dict[3])
print(Dict)


# In[21]:


#b. Add ‘features’ in the dictionary
Dict[4]="features"
print(Dict)


# In[23]:


#c. Fetch the ‘data’ element from the dictionary and display it in the output
print('Fetching the sictionary element', Dict[1])


# 3. Create a tuple and add these elements 1, 2, 3, apple, mango in my_tuple.
# 

# In[25]:


my_tuple=(1,2,3,'apple','mango')


# In[38]:


#4. Create another tuple named numeric_tuple consisting of only integer values 10, 20, 30, 40, 50:
numeric_tuple=(10,20,30,40,50)

#a. Find the minimum value from the numeric_tuple
min(numeric_tuple)



# In[37]:


#b. Concatenate my_tuple with numeric_tuple and store the result in r1
r1=my_tuple+numeric_tuple
print(r1)


# In[36]:


#c. Duplicate the tuple named my_tuple 2 times and store that in ‘newdupli’
newdupli=((my_tuple)*2)
print(newdupli)


# In[47]:


#5. Create 2 sets with the names set1 and set2, where set1 contains {1,2,3,4,5} and set2 contains {2,3,7,6,1}

set1={1,2,3,4,5}
set2={2,3,7,6,1}

#Perform the below operation:
#a. set1 union set2
set1.union(set2)


# In[44]:


#b. set1 intersection set2
set1.intersection(set2)


# In[48]:


#c. set1 difference set2
set1.difference(set2)


# In[ ]:




