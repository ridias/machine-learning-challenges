import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("dataset.csv")

print(data.isnull().sum())
print(data.isna().sum())

le_type = LabelEncoder()

data['type'] = le_type.fit_transform(data['type'])

X = data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = data['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: { accuracy }')