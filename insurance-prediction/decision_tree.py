import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: { accuracy }')