from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("BRCA.csv")

print(data.info())

print(data.isnull().sum())

data = data.dropna(subset=['Patient_Status'])

print(data.isnull().sum())
print(data.isna().sum())

le_gender = LabelEncoder()
le_age = LabelEncoder()
le_tumor_stage = LabelEncoder()
le_histology = LabelEncoder()
le_er = LabelEncoder()
le_pr = LabelEncoder()
le_her2 = LabelEncoder()
le_surgery_type = LabelEncoder()
le_status = LabelEncoder()

data['Gender'] = le_gender.fit_transform(data['Gender'])
data['Age'] = le_age.fit_transform(data['Age'])
data['Tumour_Stage'] = le_tumor_stage.fit_transform(data['Tumour_Stage'])
data['Histology'] = le_histology.fit_transform(data['Histology'])
data['ER status'] = le_er.fit_transform(data['ER status'])
data['PR status'] = le_pr.fit_transform(data['PR status'])
data['HER2 status'] = le_her2.fit_transform(data['HER2 status'])
data['Surgery_type'] = le_surgery_type.fit_transform(data['Surgery_type'])
data['Patient_Status'] = le_status.fit_transform(data['Patient_Status'])

X = data[['Gender', 'Age', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']]
y = data['Patient_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo SVM: {:.2f}".format(accuracy))