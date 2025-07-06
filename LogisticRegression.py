from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

logReg = LogisticRegression()

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

X_test = test["x"].values.reshape(-1, 1)
y_test = test["y"].values
X_train = train["x"].values.reshape(-1, 1)
y_train = train["y"].values

X_test,y_test,X_train,y_train = train_test_split(X,y,test_size = 0.3, random_state = 42)

logReg.fit(X_train,y_train)

y__pred = logReg.predict(X_test)

y_pred_prob = logReg.predict_proba(X_test)[:,1]

print(y_pred_prob[0])