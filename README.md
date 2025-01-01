# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Problem Definition

Define the input variable (X) and output variable (Y).

2.Initialize Parameters

Set initial values for the weights (θ₀, θ₁) and learning rate (α).

3.Define Cost Function

Use the Mean Squared Error (MSE) as the cost function

4.Repeat Until Convergence

Continue updating weights until the cost function converges (stops decreasing significantly).

5.Prediction

Use the final weights (θ₀, θ₁) to make predictions.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: vishal.v
RegisterNumber:24900179  
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv(r'C:\Users\admin\Downloads\50_Startups.csv')
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")
```

## Output:
![Screenshot 2024-11-15 225927](https://github.com/user-attachments/assets/c3014d65-0946-4d98-a803-5662f44bad12)
![Screenshot 2024-11-15 225945](https://github.com/user-attachments/assets/b025adbb-084f-477c-b51c-366bb4e80483)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
