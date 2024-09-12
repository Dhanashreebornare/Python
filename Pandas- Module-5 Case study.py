#!/usr/bin/env python
# coding: utf-8

# # Pandas Module-5 Case study

# In[ ]:


#write a function that takes start and end of range returns a pandas series object containing numbers within that range
#in case the user does not pass start or end or both should return default to 1 and 10 respectively 


# In[7]:


import pandas as pd

def gen_range_series(start=1, end=10):
    # Generate a Series with numbers within the specified range
    num_series = pd.Series(range(start, end + 1))
    return num_series

# Test the function
start_range = 5
end_range = 13
result_series = generate_range_series(start_range, end_range)
print(result_series)


# In[17]:


#1. create a method that takes n Numpy arrays of the same dimensions, sums them and returns the answer

import numpy as np

def sum_arrays(*arrays):
    # Check if at least one array is provided
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")

    # Check if all arrays have the same shape
    reference_shape = arrays[0].shape
    for array in arrays:
        if array.shape != reference_shape:
            raise ValueError("All arrays must have the same dimensions.")

    # Sum the arrays element-wise
    result = np.zeros(reference_shape)
    for array in arrays:
        result += array

    return result

# Test the function
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])
array3 = np.array([[9, 10], [11, 12]])
result = sum_arrays(array1, array2, array3)
print(result)


# #2.create a function that takes in two lists named keys and values as arguments
# keys would be strings and contain n string values. Values  would be a list containing n lists. The methods should return a new pandas Dataframe with keys as column names and values as their corresponding values

# In[18]:


import pandas as pd

def create_dataframe(keys, values):
    # Check if the lengths of keys and values match
    if len(keys) != len(values):
        raise ValueError("Number of keys and number of value lists must be the same.")

    # Create a dictionary to hold the data
    data_dict = {keys[i]: values[i] for i in range(len(keys))}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    return df

# Test the function
keys = ['Name', 'Middle name', 'Surname']
values = [['Dhanashree', 'Snehal', 'Usha'], ['Daulat', 'Gaurav', 'Vijay'], ['Mahale', 'Abhale', 'Bornare']]

result_df = create_dataframe(keys, values)
print(result_df)






# #3.Create a function that concatenates two dataframes. Use a previously created function to create two dataframes and pass them as parameters. Make sure that the indexes are reset before returning

# In[32]:


import pandas as pd
def create_dataframe(keys, values):
    if len(keys) != len(values):
        raise ValueError("Number of keys and number of value lists must be the same.")

    # Create a dictionary to hold the data
    data_dict = {keys[i]: values[i] for i in range(len(keys))}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    return df
def concatenate_df(df1, df2):
    # Concatenate the two DataFrames vertically
        concatenated_df = pd.concat([df1, df2], ignore_index=True)
        return concatenated_df

# Create two DataFrames using the create_dataframe function
keys1 = ['Name', 'Age']
values1 = [['Dhanashree', 'Amol', 'Parth'], [33, 39, 8]]
df1 = create_dataframe(keys1, values1)

keys2 = ['Name', 'Age']
values2 = [['Usha', 'Snehal'], [59, 35]]
df2 = create_dataframe(keys2, values2)

# Concatenate the two DataFrames
result_df = concatenate_df(df1, df2)
print(result_df)


# #4. write code to load data from cars,csv inot a dataframe and print its deails. Details like- 'count', 'mean', 'std', 'min', '25%', 50%, 75%', 'max'.

# In[34]:


import pandas as pd
df=pd.read_csv('D:\Intellipaat\Python\PANDAS\cars.csv')
df


# In[36]:


summary_stats = df.describe()
summary_stats


# #5. write a method that will take  a column name as argument and return the name of the column with which  the given column has the highest correlation.

# In[38]:


import pandas as pd

def find_highest_correlation(df, target_column):
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in the DataFrame.")

    # Check if the target column is numeric
    if df[target_column].dtype not in [int, float]:
        raise ValueError(f"Target column '{target_column}' must be numeric.")

    # Calculate correlations only for numeric columns
    numeric_columns = df.select_dtypes(include=[int, float]).columns
    numeric_columns = numeric_columns.drop(target_column)
    
    correlations = df[numeric_columns].corrwith(df[target_column])

    # Find the column with the highest correlation
    highest_correlation_column = correlations.idxmax()

    return highest_correlation_column

# Load data from a CSV file into a DataFrame
data_path = 'D:/Intellipaat/Python/PANDAS/cars.csv'  # Update the path as needed
df = pd.read_csv(data_path)

# Choose a target column for correlation analysis
target_column = 'mpg'

# Find the column with the highest correlation to the target column
highest_correlation = find_highest_correlation(df, target_column)
print(f"The column with the highest correlation to '{target_column}' is '{highest_correlation}'.")


# In[ ]:




