import numpy as np
from sklearn.linear_model import LinearRegression


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])


print("x:", x)
print("y:", y)


model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination (R^2):', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


y_pred_manual = model.intercept_ + model.coef_ * x
print('manual predicted response:', y_pred_manual, sep='\n')


new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept (new model):', new_model.intercept_)
print('slope (new model):', new_model.coef_)

x_new = np.arange(5).reshape((-1, 1))
y_new = model.predict(x_new)
print("New x values:", x_new)
print("Predicted y values for new x:", y_new)
