import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("dataset.csv")

print(dataset.info())
print(dataset.isnull().sum())
print(dataset.isna().sum())

X = dataset[['total_bill', 'size']]
y = dataset['tip']

plt.scatter(dataset['size'], y)
plt.xlabel("Número de personas")
plt.ylabel("Tip")
plt.title("Número de personas vs tip")
plt.show()

plt.scatter(dataset['total_bill'], y)
plt.xlabel("Factura")
plt.ylabel("Tip")
plt.title("Factura vs tip")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(model.coef_)
print(model.intercept_)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error Cuadrático Medio (MSE):", mse)
print("Coeficiente de Determinación (R^2):", r2)
print("Score: ", model.score(X_test, y_test))

plt.scatter(y_test, y_pred)
plt.xlabel("Real sales")
plt.ylabel("Predicted sales")
plt.title("Real sales vs Predicted sales")
plt.show()