from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("dataset.csv")

print(dataset.info())

print(dataset.isnull().sum())
print(dataset.isna().sum())

dataset = dataset.dropna()
print(dataset.isna().sum())

plt.scatter(dataset["Total Price"], dataset["Base Price"], dataset["Units Sold"])
plt.show()

X = dataset[['Total Price', "Base Price"]]
y = dataset["Units Sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(model.coef_)
print(model.intercept_)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprime los resultados
print("Error Cuadrático Medio (MSE):", mse)
print("Coeficiente de Determinación (R^2):", r2)
print("Score: ", model.score(X_test, y_test))