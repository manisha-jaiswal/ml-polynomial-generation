#difference between normal linear fit and parabolic fit using polynomial regression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd
#generating random numbers for equation y=X^3+100+e
np.random.seed(42)
n_samples=100
X=np.linspace(0,10,100)
rng=np.random.randn(n_samples)*100
y=X**3+rng+100
#plot the data in scatter plot
plt.figure(figsize=(10,8));
plt.scatter(X,y);
plt.show()


#performing linear regression on the X and y data 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr=LinearRegression()
lr.fit(X.reshape(-1,1),y);
model_pred=lr.predict(X.reshape(-1,1))
#prediction plot will be linear
plt.figure(figsize=(10,8));
plt.scatter(X,y);
plt.plot(X,model_pred);
#r2 score will be low
print(r2_score(y,model_pred))
plt.show()


#performing linear regression on the X and y data with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X.reshape(-1,1))
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y.reshape(-1,1))
y_pred=lin_reg_2.predict(X_poly)
#prediction plot will be parabolic
plt.figure(figsize=(10,8));
plt.scatter(X,y);
plt.plot(X,y_pred);
#r2 score will be high (best fit)
print(r2_score(y,y_pred))
plt.show()
