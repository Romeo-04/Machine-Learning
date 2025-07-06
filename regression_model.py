import numpy as np
import pandas as pd 

test = pd.read_csv("test.csv")

# Create X from the radio column's values
X = test["x"].values

# Create y from the sales column's values
y = test["y"].values

# Reshape X
X = X.reshape(-1,1)

# Check the shape of the features and targets
print(X.shape,y.shape)