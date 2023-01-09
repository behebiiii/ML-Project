import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

id = 'id'
target = 'failure'

rt = "C:/Users/steve/Desktop/機器學習/final project"
os.chdir(rt)
print(os.getcwd())


train_data = pd.read_csv('./input/train.csv', index_col='id')
test_data = pd.read_csv('./input/test.csv', index_col='id')
submission = pd.read_csv('./input/sample_submission.csv')
# print(f'train {train_df.shape}, test {test_df.shape}')


Y_column = ['failure']

X_columns_attribute = [x for x in train_data.columns.values if x.startswith('attribute_')]
X_columns_loading = ['loading']
X_columns_measurement = [x for x in train_data.columns.values if x.startswith('measurement_')]
X_columns = X_columns_attribute + X_columns_loading + X_columns_measurement


# print(X_columns_attribute)
# print(X_columns_loading)
# print(X_columns_measurement)

X_columns_categorical = X_columns_attribute
X_columns_int = [x for x in X_columns_measurement if train_data[x].dtype == "int64"]
X_columns_float = [x for x in X_columns if train_data[x].dtype == "float"]

# print(X_columns)
X_columns.sort() == (X_columns_categorical + X_columns_int + X_columns_float).sort()
print(X_columns.sort() == (X_columns_categorical + X_columns_int + X_columns_float).sort())
# print(train_data.info())
# print(X_columns_int)
# print(X_columns_float)
# print(X_columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train_data[X_columns], train_data[Y_column],
    test_size=0.20, random_state=0, stratify=train_data[Y_column]
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

int_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

float_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, X_columns_categorical),
        ('int', int_transformer, X_columns_int),
        ('float', float_transformer, X_columns_float)
    ]
)


# Logistic Regression with cross validation

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
logistic_regression_model = LogisticRegressionCV(random_state=0, cv=5, scoring="roc_auc")

# Pipeline

logistic_regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', logistic_regression_model)
])


# Train
print(y_train)
logistic_regression_pipeline.fit(X_train, y_train.values.ravel())



logistic_regression_best_hyper_paras = {
    'C': logistic_regression_pipeline.named_steps['model'].C_[0],
    #'l1_ratio': logistic_regression_pipeline.named_steps['model'].l1_ratio_[0]
}

print(logistic_regression_best_hyper_paras)



y_predict_log_regress = logistic_regression_pipeline.predict(X_test)
y_pred_lr = pd.Series(
    logistic_regression_pipeline.predict_proba(test_data[X_columns])[:, 1],
)


output = pd.DataFrame({'id': test_data.index,
                       'failure': y_pred_lr})
output.to_csv('submission.csv', index=False)

cm_log_regress = confusion_matrix(y_test, y_predict_log_regress)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log_regress, display_labels=logistic_regression_pipeline.classes_)
disp.plot()


from sklearn.metrics import f1_score, accuracy_score
f1_log_regress = f1_score(y_test, y_predict_log_regress)
accuracy_log_regress = accuracy_score(y_test, y_predict_log_regress)
print(f"Logistic Model: F1 Score = {f1_log_regress :0.4f}, Accuracy = {accuracy_log_regress :0.4f}")