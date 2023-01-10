import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

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


# Read data  ， You may need to change the path
rt = "C:/Users/steve/Desktop/機器學習/final_project"
os.chdir(rt)

train_data = pd.read_csv('./input/train.csv', index_col='id')
test_data = pd.read_csv('./input/test.csv', index_col='id')

# divide columns
Y_column = ['failure']
X_columns = [x for x in train_data.columns.values if x != 'id' and x != 'product_code' and x != 'failure' ]

X_train, X_test, y_train, y_test = train_test_split(
    train_data[X_columns], train_data[Y_column],
    test_size=0.20, random_state=0, stratify=train_data[Y_column]
)

# Reference : https://www.kaggle.com/code/maxsarmento/lb-0-58978-standing-on-the-shoulder-of-giants?scriptVersionId=102785631
# adding more hiddden features
create_features(X_train)
create_features(X_test)
create_features(train_data)
create_features(test_data)

X_columns = [
    'loading',
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
] 


#資料前處理，將所有特徵標準化
scaler = StandardScaler()


# one-hot encoded
Xt = pd.get_dummies(X_train, columns=['attribute_0'])
Xt.drop(["attribute_1"], axis=1, inplace=True)

# Fill NA/NaN values using the specified method.
for feature in X_columns:
    # print("checked missing data(NAN mount):",len(np.where(np.isnan(Xt[feature]))[0]))
    if X_train[feature].dtype == "float":
        Xt[feature].fillna((X_train[feature].mean()), inplace=True)
    if X_train[feature].dtype == "object":
        Xt[feature].fillna((X_train[feature].mode()), inplace=True)
    if X_train[feature].dtype == "int":
        Xt[feature].fillna((X_train[feature].median()), inplace=True)
    # print("checked missing data(NAN mount):",len(np.where(np.isnan(Xt[feature]))[0]))

#標準化
Xt= scaler.fit_transform(Xt)

# print('資料集 X 的平均值 : ', X_train.mean(axis=0))
# print('資料集 X 的標準差 : ', X_train.std(axis=0))

# print('\nStandardScaler 縮放過後訓練集的平均值 : ', Xte.mean(axis=0))
# print('StandardScaler 縮放過後訓練集的標準差 : ', Xte.std(axis=0))


# 對 X_test 做同樣處理
Xte = pd.get_dummies(X_test, columns=['attribute_0'])
Xte.drop(["attribute_1"], axis=1, inplace=True)

for feature in X_columns:
        if X_train[feature].dtype == "float":
            Xte[feature].fillna((X_train[feature].mean()), inplace=True)
        if X_train[feature].dtype == "object":
            Xte[feature].fillna((X_train[feature].mode()), inplace=True)
        if X_train[feature].dtype == "int":
            Xte[feature].fillna((X_train[feature].median()), inplace=True)
    
Xte= scaler.transform(Xte)



# 對 test_data 做同樣處理

test = pd.get_dummies(test_data, columns=['attribute_0'])
test.drop(["product_code","attribute_1"], axis=1, inplace=True)

for feature in X_columns:
        test[feature].fillna((X_train[feature].mean()), inplace=True)
        if X_train[feature].dtype == "float":
            test[feature].fillna((X_train[feature].mean()), inplace=True)
        if X_train[feature].dtype == "object":
            test[feature].fillna((X_train[feature].mode()), inplace=True)
        if X_train[feature].dtype == "int":
            test[feature].fillna((X_train[feature].median()), inplace=True)
    
test= scaler.transform(test)


# Use gridsearch to tune parameters
model = LogisticRegression()

# Load the Model back from file
Pkl_Filename = "Model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    model_re = pickle.load(file)

param_grid = [    
    {'penalty' : ['l1'],
    'C' : np.arange(0, 1, 0.01),
    'solver' : ['liblinear'],
    'max_iter' : [100, 500, 1000],
    
    }
]

# model_re = GridSearchCV(model, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# model_re.fit(Xt,y_train.values.ravel())
# accuracy = model_re.score(Xte, y_test)
# print("accuracy {:.2f}".format(accuracy))


# predict
probs_test = model_re.predict_proba(test)

# Save the Modle to file in the current working directory
# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(model_re, file)

# output
df_submission = pd.DataFrame({"id": test_data.index,
                        "failure": probs_test[:,1]})
df_submission.to_csv("109550035.csv", index=False)
