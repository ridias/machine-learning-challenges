# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Youtube01-Psy.csv")

X = dataset['CONTENT']
y = dataset['CLASS']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = BernoulliNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del clasificador Naïve Bayes Multinomial: {accuracy * 100:.2f}%")
