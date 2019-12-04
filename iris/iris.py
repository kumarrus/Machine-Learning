# %% Read data

import pandas as pd

DATA_LOCATION = 'iris\\data\\iris.data'

data = pd.read_csv(DATA_LOCATION, header=None, names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'PlantClass'])
data.head(5)
data.describe()

# %% split train and test data
from sklearn.model_selection import train_test_split

y = data.PlantClass
X = data
data.drop(['PlantClass'], axis=1, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.2)

# %% compare models

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('DTC', DecisionTreeClassifier()),
    ('KN', KNeighborsClassifier()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('GNB', GaussianNB()),
    ('SVC', SVC(gamma='auto')),
]

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_score = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_score)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_score.mean(), cv_score.std()))

# %% Select best model and predict reults
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

iris_model = SVC(gamma='auto')
iris_model.fit(X_train, y_train)
predictions = iris_model.predict(X_val)

# Evaluate predictions
print(accuracy_score(y_val, predictions))
print(confusion_matrix(y_val, predictions))
print(classification_report(y_val, predictions))
