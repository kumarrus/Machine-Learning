#%% Read Data
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('kaggle/titanic/data/train.csv', index_col='PassengerId')
test_data = pd.read_csv('kaggle/titanic/data/test.csv', index_col='PassengerId')

#%% Cleanup data

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

# print(train_data.count())
# print(train_data.dtypes)
print(train_data.isna().sum())

def transformTickets(dataframe, y=None):
    # Ticket column has the format "prefix number". Separate into two columns
    prefix = []
    number = []
    for tkt in dataframe['Ticket']:
        first = tkt
        second = None
        if tkt:
            matches = tkt.split()
            if matches[-1].isnumeric():
                second = matches[-1]
                first = tkt.replace(matches[-1], '').replace(' ', '')

        prefix.append(first)
        number.append(second)

    dataframe['TicketPrefix'] = prefix
    dataframe['TicketNum'] = number
    return dataframe

def dropUnnecessaryData(dataframe, y=None):
    # Columns will invalid data:
    # Age         177
    # Cabin       687
    # Embarked      2

    transformTickets(dataframe)
    # Too many NaNs in Cabin data to be useful (687 nulls in training data)
    dataframe.drop(['Name', 'Cabin', 'Ticket', 'TicketPrefix'], axis=1, inplace=True)
    return dataframe

# Too many NaNs in Cabin data to be useful (687 nulls)
unused_features = ['Name', 'Cabin', 'Ticket']

numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Embarked', 'Sex', 'Pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

composer = FunctionTransformer(dropUnnecessaryData, validate=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ])

y = train_data.Survived
train_data.drop(['Survived'], axis=1, inplace=True)
X = train_data
res = composer.transform(X)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Train & test model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

titanic_model = XGBClassifier(n_jobs=4, random_state=1)
titanic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', titanic_model)
])
titanic_pipeline.fit(X_train, y_train)
predictions = titanic_pipeline.predict(X_valid)

# Evaluate predictions
print(accuracy_score(y_valid, predictions))
print(confusion_matrix(y_valid, predictions))
print(classification_report(y_valid, predictions))

#%% Get competition output
X_test = composer.transform(test_data)

titanic_pipeline.fit(X, y)
test_preds = titanic_pipeline.predict(X_test)

# save predictions in format used for competition scoring
output = pd.DataFrame({'PassengerId': X_test.index,
                       'Survived': test_preds})
output.to_csv('kaggle/titanic/submission.csv', index=False)

#%% Evaluate different models

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
#
# def get_accuracy(model):
#     """Return the accuracy over 3 CV folds of the given model.
#
#     Keyword argument:
#     model -- the model to get accuracy for
#     """
#
#     my_pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('model', model)
#     ])
#
#     kfold = StratifiedKFold(n_splits=3, random_state=1)
#     scores = cross_val_score(my_pipeline, X, y, cv=3, scoring='accuracy')
#
#     return scores.mean()
#
#
# models = [
#     ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
#     ('DTC', DecisionTreeClassifier()),
#     ('RFC', RandomForestClassifier(n_estimators=100, random_state=1)),
#     ('KN', KNeighborsClassifier()),
#     ('LDA', LinearDiscriminantAnalysis()),
#     ('GNB', GaussianNB()),
#     ('SVC', SVC(gamma='auto', random_state=1)),
#     ('XGB', XGBClassifier(n_jobs=4, random_state=1)),
#     ('XGB1K', XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=1)),
# ]
#
# for name, model in models:
#     cv_accuracy = get_accuracy(model)
#     print('%s: %f' % (name, cv_accuracy))
