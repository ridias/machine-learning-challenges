from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Social_Network_Ads.xls")

print(dataset.info())
print(dataset.isnull().sum())
print(dataset.isna().sum())

le_gender = LabelEncoder()

dataset['Gender'] = le_gender.fit_transform(dataset['Gender'])

colors = {0:'red', 1:'green'}

plt.scatter(x=dataset['Age'], y=dataset['EstimatedSalary'], c=[colors[i] for i in dataset['Purchased']])
plt.xlabel("Age")
plt.ylabel("Estimated salary")
plt.show()

X = dataset[['Age', 'Gender', 'EstimatedSalary']]
y = dataset['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(report)

print("Matrix confusion")
cm = confusion_matrix(y_test, y_pred)
print(cm)
