from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((m,1))

#Creating model
reg = LinearRegression()

#Fitting training data
reg = reg.fit(X,Y)

# Y prediction
Y_pred = reg.predict(X)

#Calculating the R2 score
r2_score = reg.score(X,Y)

print(r2_score)
