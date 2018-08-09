import pandas as pd 
pd.__version__ 
import matplotlib.pyplot as plt 
import numpy as np 

# The primary data structures in pandas are implemented as two classes: 
#  * DataFrame, which you can imagine as a relational data table, with rows and named columns.
#  * Series, which is a single column. A DataFrame contains one or more Series and a name for each Series. 

cities = pd.Series(['San Francisco', 'San Jose', 'Sacramento']) 
print(cities)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento']) 
population = pd.Series([852469, 1015785, 485199])

dataset = pd.DataFrame({'City name': city_names, 'Population': population}) 

print(dataset) 

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())

print(california_housing_dataframe.head())

plt.show(california_housing_dataframe.hist('housing_median_age'))

print(type(dataset['City name']))
print(dataset['City name'])

print(type(dataset['City name']))
print(dataset['City name'][1])

print(type(dataset[0:2]))
print(dataset[0:2])

print(population / 1000)
print(np.log(population))
dataset['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
dataset['Population density'] = dataset['Population'] / dataset['Area square miles']
print(dataset)

dataset['Is wide and has saint name'] = (dataset['Area square miles'] > 50) & dataset['City name'].apply(lambda name: name.startswith('San'))

print(dataset)