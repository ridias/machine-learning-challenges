from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
random_forest.fit(X_train, y_train)

print(f"Otra manera de mirar la precision del modelo: ", random_forest.score(X_test, y_test))
# Realizar predicciones en el conjunto de prueba
y_pred = random_forest.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Generar un informe de clasificación
report = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(report)

print("Matrix confusion")
cm = confusion_matrix(y_test, y_pred)
print(cm)