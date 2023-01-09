import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

rt = "C:/Users/steve/Desktop/機器學習/final project"
os.chdir(rt)
print(os.getcwd())


train_data = pd.read_csv('./input/train.csv', index_col='id')
test_data = pd.read_csv('./input/test.csv', index_col='id')

Y_column = ['failure']

X_columns_attribute = [x for x in train_data.columns.values if x.startswith('attribute_')]
X_columns_loading = ['loading']
X_columns_measurement = [x for x in train_data.columns.values if x.startswith('measurement_')]
X_columns = X_columns_attribute + X_columns_loading + X_columns_measurement


# print(X_columns_attribute)
# print(X_columns_loading)
# print(X_columns_measurement)
print(X_columns)
X_columns_categorical = X_columns_attribute
X_columns_int = [x for x in X_columns_measurement if train_data[x].dtype == "int64"]
X_columns_float = [x for x in X_columns if train_data[x].dtype == "float"]

X_train, X_test, y_train, y_test = train_test_split(
    train_data[X_columns], train_data[Y_column],
    test_size=0.20, random_state=0, stratify=train_data[Y_column]
)


def create_features(df):
    df['attribute_2*3'] = df['attribute_2'] * df['attribute_3']

    df['measurement_0 / loading'] = df['measurement_0'] / df['loading']
    df['measurement_1 / loading'] = df['measurement_1'] / df['loading']
    df['measurement_2 / loading'] = df['measurement_2'] / df['loading']
    df['measurement_3_to_16_mean / loading'] = df[[f"measurement_{x:d}" for x in range(3,17)]].mean(axis='columns') / df['loading']
    df['measurement_17 / loading'] = df['measurement_17'] / df['loading']

    df['m3_missing'] = df['measurement_3'].isnull().astype(np.int8)
    df['m5_missing'] = df['measurement_5'].isnull().astype(np.int8)
    df['m_3*5_missing'] = df['m3_missing'] * df['m5_missing']
    meas_gr1_cols = [f"measurement_{i:d}" for i in list(range(3, 5)) + list(range(9, 17))]
    df['meas_gr1_avg'] = np.mean(df[meas_gr1_cols], axis=1)
    df['meas_gr1_std'] = np.std(df[meas_gr1_cols], axis=1)

    meas_gr2_cols = [f"measurement_{i:d}" for i in list(range(5, 9))]
    df['meas_gr2_avg'] = np.mean(df[meas_gr2_cols], axis=1)
    
    df['meas17/meas_gr2_avg'] = df['measurement_17'] / df['meas_gr2_avg']


create_features(X_train)
create_features(X_test)
create_features(train_data)
create_features(test_data)

X_columns = [
    'loading',
    'attribute_0',
    'attribute_1',
    'attribute_2*3',
    'measurement_3', 'measurement_4', 'measurement_5',
            'measurement_6', 'measurement_7', 'measurement_8', 'measurement_9',
            'measurement_10', 'measurement_11', 'measurement_12', 'measurement_13',
            'measurement_14', 'measurement_15', 'measurement_16', 'measurement_17', 
    'measurement_0 / loading',
    'measurement_1 / loading',
    'measurement_2 / loading',
    'measurement_3_to_16_mean / loading',
    'measurement_17 / loading',
    
    'm3_missing',
    'm5_missing',
    #'m_3*5_missing',
    #'meas_gr1_avg',
    #'meas_gr1_std', 
    #'meas17/meas_gr2_avg'
] 

num_columns=['loading', 'measurement_3', 'measurement_4', 'measurement_5',
            'measurement_6', 'measurement_7', 'measurement_8', 'measurement_9',
            'measurement_10', 'measurement_11', 'measurement_12', 'measurement_13',
            'measurement_14', 'measurement_15', 'measurement_16', 'measurement_17']


# X_train[X_columns].info()

preprocessor = ColumnTransformer(transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), [x for x in X_columns if train_data[x].dtype == "object"]),
        ('int', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), [x for x in X_columns if train_data[x].dtype == "int"]),
        ('float', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
            ('scaler', StandardScaler())
        ]), [x for x in X_columns if train_data[x].dtype == "float"])
    ])


fe_logistic_regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegressionCV(
        random_state=0, cv=5,
        scoring="roc_auc",
        penalty='elasticnet', 
        l1_ratios=np.arange(0, 1.01, 0.1), 
        Cs=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        solver='saga',
        tol = 1e-3, max_iter=5000
    ))
])

fe_logistic_regression_pipeline.fit(X_train, y_train.values.ravel())

fe_logistic_regression_best_hyper_paras = {
    'C': fe_logistic_regression_pipeline.named_steps['model'].C_[0],
    'l1_ratio': fe_logistic_regression_pipeline.named_steps['model'].l1_ratio_[0]
}

print(fe_logistic_regression_best_hyper_paras)

y_predict_fe_log_regress = fe_logistic_regression_pipeline.predict(X_test)

y_predict_proba_fe_regress = fe_logistic_regression_pipeline.predict_proba(X_test)[:,1]
roc_score_fe_regress = roc_auc_score(y_test, y_predict_proba_fe_regress)
print(f"Logistic Regression Model: ROC Score = {roc_score_fe_regress : .4f}")


selected_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(random_state=0,
        penalty='elasticnet',
        C=fe_logistic_regression_best_hyper_paras['C'],
        l1_ratio=fe_logistic_regression_best_hyper_paras['l1_ratio'],
        solver='saga',
        tol = 1e-3, max_iter=5000
    ))
])

selected_model.fit(train_data[X_columns], train_data[Y_column].values.ravel())

test_data_predict_prob = selected_model.predict_proba(test_data[X_columns])[:, 1]

output = pd.DataFrame({'id': test_data.index,
                       'failure': test_data_predict_prob})
output.to_csv('submission.csv', index=False)

# #資料前處理，將所有特徵標準化
# # scaler = StandardScaler()
# scaler = RobustScaler()
# #RobustScaler 可以有效的縮放帶有outlier的數據，透過Robust如果數據中含有異常值在縮放中會捨去。

# Xt = pd.get_dummies(X_train, columns=['attribute_0'])
# Xt.drop(["attribute_1"], axis=1, inplace=True)
# # print(Xt.info())
# num_columns=['loading', 'measurement_3', 'measurement_4', 'measurement_5',
#             'measurement_6', 'measurement_7', 'measurement_8', 'measurement_9',
#             'measurement_10', 'measurement_11', 'measurement_12', 'measurement_13',
#             'measurement_14', 'measurement_15', 'measurement_16', 'measurement_17']

# # Fill NA/NaN values using the specified method.
# for feature in num_columns:
#     # print("checked missing data(NAN mount):",len(np.where(np.isnan(Xt[feature]))[0]))
#     Xt[feature].fillna((X_train[feature].mean()), inplace=True)
#     # print("checked missing data(NAN mount):",len(np.where(np.isnan(Xt[feature]))[0]))

# #標準化
# Xt= scaler.fit_transform(Xt)
            


# Xte = pd.get_dummies(X_test, columns=['attribute_0'])
# Xte.drop(["attribute_1"], axis=1, inplace=True)
# for feature in num_columns:
#         Xte[feature].fillna((X_train[feature].mean()), inplace=True)
    
# Xte= scaler.transform(Xte)

# # print('資料集 X 的平均值 : ', X_train.mean(axis=0))
# # print('資料集 X 的標準差 : ', X_train.std(axis=0))

# # print('\nStandardScaler 縮放過後訓練集的平均值 : ', Xte.mean(axis=0))
# # print('StandardScaler 縮放過後訓練集的標準差 : ', Xte.std(axis=0))


# test = pd.get_dummies(test_data, columns=['attribute_0'])
# # print(test)
# test.drop(["product_code","attribute_1"], axis=1, inplace=True)

# for feature in num_columns:
#         test[feature].fillna((X_train[feature].mean()), inplace=True)
    
# test= scaler.transform(test)


# clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.8, C=0.007, tol = 1e-2, solver='saga', max_iter=1000, random_state=i)


# # clf =  LogisticRegressionCV(
# #         random_state=0, cv=5,
# #         scoring="roc_auc",
# #         penalty='elasticnet', 
# #         l1_ratios=np.arange(0, 1.01, 0.1), 
# #         Cs=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
# #         solver='saga',
# #         tol = 1e-3, max_iter=5000
# #     )

# clf.fit(Xt,y_train)
# accuracy = clf.score(Xte, y_test)
# print("accuracy {:.2f}".format(accuracy))
# auc = roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
# print("accuracy {:.2f}".format(auc))


# probs_test = clf.predict_proba(test)
# df_submission = pd.DataFrame({"id": test_data.index,
#                         "failure": probs_test[:,1]})
# df_submission.to_csv("submission.csv", index=False)
