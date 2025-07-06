import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

test = pd.read_csv("test.csv")

X = test["x"].values
y = test["y"].values

X = X.reshape(-1,1)

reg = LinearRegression()
reg.fit(X,y)

predictions = reg.predict(X)

plt.scatter(X,y,color = "green")
plt.plot(X,predictions,color = "red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sample Linear Regression")

plt.show()