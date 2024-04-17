import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Display the first few rows of the dataset
print(df.head())

# Calculate the average honey production per year
prod_per_year = df.groupby('year')['totalprod'].mean().reset_index()
print(prod_per_year)

# Prepare the data for linear regression
x = prod_per_year['year'].values.reshape(-1, 1)
y = prod_per_year['totalprod']

# Visualize the data using a scatter plot
plt.scatter(x, y)
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.title('Average Honey Production per Year')
plt.show()

# Create a linear regression model and fit it to the data
regr = LinearRegression()
regr.fit(x, y)

# Print the coefficients and intercept of the linear regression model
#print(regr.coef_[0])
#print(regr.intercept_)

# Predict honey production for the years in the dataset
y_predict = regr.predict(x)

# Plot the regression line
plt.plot(x, y_predict)
plt.xlabel('Year')
plt.ylabel('Predicted Total Production')
plt.title('Linear Regression Model')
plt.show()

# Predict future honey production
x_future = np.arange(2013, 2050).reshape(-1, 1)
future_predict = regr.predict(x_future)

# Plot the predicted future production
plt.plot(x_future, future_predict)
plt.xlabel('Year')
plt.ylabel('Predicted Total Production')
plt.title('Future Honey Production Prediction')
plt.show()
