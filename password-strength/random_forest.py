import re 

def countLowLetters(text):
    matches = re.findall("[a-z]", text)
    return len(matches)

def countCapletters(text):
    matches = re.findall("[A-Z]", text)
    return len(matches)

def countDigits(text):
    matches = re.findall("[0-9]", text)
    return len(matches)

def countSpecialChars(text): 
    matches = re.findall("[\W_]", text)
    return len(matches)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

file = open('data.csv', 'r')
lines = file.readlines()

dataset = []
count = 0
for i in range(1, len(lines)):
    line = lines[i]
    strength = line[-2]
    password = line[0:-3]

    lower = countLowLetters(password)
    upper = countCapletters(password)
    digits = countDigits(password)
    specialChars = countSpecialChars(password)
    length = len(password)

    dataset.append([lower, upper, digits, specialChars, length, int(strength)])
    
my_dataset = np.array(dataset)
df = pd.DataFrame(my_dataset, columns=['num_lower', 'num_upper', 'num_digits', 'num_special', 'length', 'strength'])

print(df.info())

print(df.isna().sum())

X = df.drop('strength', axis=1)
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(X_train, y_train)
score = random_forest.score(X_test, y_test)
print(f'Score: { score }')

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Generar un informe de clasificación
report = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(report)

print("Matrix confusion")
cm = confusion_matrix(y_test, y_pred)
print(cm)