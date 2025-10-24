# co2-for-cars
# Obtaining co2 different  cars by using engine size.
# co2-for-cars

# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# we to a co2.csv file 
# read the csv file
df = pd.read_csv('co2.csv')
df.describe()

# # This code creates a count plot of the 'out1' column in the dataframe.
sns.countplot(x='out1',data=df)

# create a  hit  map 
plt.subplots(figsize=(9,9))
sns.heatmap(df.corr(),annot=True)

# x = df.drop("out1",axis=1) # Drop the 'out1' column from the DataFrame and assign to x
# x = df.drop("fuelcomb",axis=1) # Drop the 'fuelcomb' column from x
# x = df.drop("cylandr",axis=1) # Drop the 'cylandr' column from x
# y = df.out1 # Select the 'out1' column as the target variable y

x = df.drop("out1",axis=1)
x = x.drop("cylandr",axis=1)
x = x.drop("fuelcomb",axis=1)
y = df.out1

# Split the data into training and testing sets
# For splite x and y as train becuase we want trained some of (x) not all
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Now as we watch we have 400 x_train 
x_train

# And also x_test  100 for after train 
x_test

# Initialize a Linear Regression model
model_reg_linear = linear_model.LinearRegression()

# Fit the linear regression model to the training data
model_reg_linear.fit(x_train,y_train)

# Predict the target variable for the test set
out_robot = model_reg_linear.predict(x_test)

out_robot

# before we called and now we call it again for udentify
x_test

# Create a scatter plot of the test data
plt.scatter(x_test,y_test, color='red')
plt.show()

# Create a scatter plot of the test data and overlay the regression line
plt.scatter(x_test,y_test, color='red')
plt.plot(x_test, out_robot, color='black', linewidth=2)
plt.show()

# Now we able import information about engine size car
# ٍExampel here is Mercedes-Benz M120 
M120  = np.array([[ 7.6]])

# we give name of the car to our model
co2 = model_reg_linear.predict(M120)

print(co2)
