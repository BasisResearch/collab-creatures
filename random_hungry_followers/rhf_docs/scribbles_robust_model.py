import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array(proximity).reshape(-1, 1)
y = np.array(how_far)

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the values using the linear model
y_pred = model.predict(X)

# Create a scatter plot of the data points
plt.scatter(proximity, how_far, label="Data Points")

# Plot the linear regression line
plt.plot(proximity, y_pred, color="red", label="Linear Model")

# Add labels and legend
plt.xlabel("proximity")
plt.ylabel("how_far")
plt.legend()

# Show the plot
plt.show()
