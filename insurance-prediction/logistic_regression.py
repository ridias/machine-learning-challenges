import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("dataset.csv")

print(dataset.info())
print(dataset.isnull().sum())
print(dataset.isna().sum())

le_employment_type = LabelEncoder()
le_graduate = LabelEncoder()
le_frequent_flyer = LabelEncoder()
le_travelled_abroad = LabelEncoder()

dataset['Employment Type'] = le_employment_type.fit_transform(dataset['Employment Type'])
dataset['FrequentFlyer'] = le_frequent_flyer.fit_transform(dataset['FrequentFlyer'])
dataset['EverTravelledAbroad'] = le_travelled_abroad.fit_transform(dataset['EverTravelledAbroad'])

X = dataset.drop(["TravelInsurance", "GraduateOrNot"], axis=1)
y = dataset['TravelInsurance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

print(y_pred)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred))