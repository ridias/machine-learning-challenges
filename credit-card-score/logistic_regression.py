from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")
print(data.head())

# ver el tipo de datos
print(data.info())

# ver si hay datos que son null por si hay que aplicar acciones para limpiar o preprocesar esos datos
print(data.isnull().sum())
# ver si hay datos que son NA por si hay que aplicar acciones para limpiar o hacer algun tipo de calculo.
print(data.isna().sum())

print(data["Credit_Score"].value_counts())

le_payment_behaviour = LabelEncoder()
le_payment_min_amount = LabelEncoder()
le_type_loan = LabelEncoder()
le_credit_mix = LabelEncoder()
le_output = LabelEncoder()

data['Credit_Score'] = le_output.fit_transform(data["Credit_Score"])
data['Payment_Behaviour'] = le_payment_behaviour.fit_transform(data["Payment_Behaviour"])
data['Payment_of_Min_Amount'] = le_payment_min_amount.fit_transform(data["Payment_of_Min_Amount"])
data['Type_of_Loan'] = le_type_loan.fit_transform(data['Type_of_Loan'])
data['Credit_Mix'] = le_credit_mix.fit_transform(data['Credit_Mix'])

X = data[["Annual_Income", "Monthly_Inhand_Salary", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Credit_Mix", "Outstanding_Debt", "Payment_of_Min_Amount", "Total_EMI_per_month", "Payment_Behaviour", "Monthly_Balance"]]

y = data['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

print(y_pred)
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:\n", classification_report(y_test, y_pred))

