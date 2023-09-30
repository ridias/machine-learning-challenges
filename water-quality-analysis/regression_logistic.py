# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


dataset = pd.read_csv("dataset.csv")

print(dataset.info())

print(dataset.isnull().sum())
print(dataset.isna().sum())

mean = dataset.mean()
dataset['ph'] = dataset['ph'].fillna(mean['ph'])
dataset['Sulfate'] = dataset['Sulfate'].fillna(mean['Sulfate'])
dataset['Trihalomethanes'] = dataset['Trihalomethanes'].fillna(mean['Trihalomethanes'])

print(dataset.isna().sum())

plt.scatter(dataset['ph'], dataset['Potability'])
plt.xlabel("ph")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Hardness'], dataset['Potability'])
plt.xlabel("Hardness")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Solids'], dataset['Potability'])
plt.xlabel("Solids")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Chloramines'], dataset['Potability'])
plt.xlabel("Chloramines")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Sulfate'], dataset['Potability'])
plt.xlabel("Sulfate")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Conductivity'], dataset['Potability'])
plt.xlabel("Conductivity")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Organic_carbon'], dataset['Potability'])
plt.xlabel("Organic_carbon")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Turbidity'], dataset['Potability'])
plt.xlabel("Turbidity")
plt.ylabel("Potability")
plt.show()

plt.scatter(dataset['Trihalomethanes'], dataset['Potability'])
plt.xlabel("Trihalomethanes")
plt.ylabel("Potability")
plt.show()

X = dataset.drop("Potability", axis=1)
y = dataset['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_test, y_test))
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred))
