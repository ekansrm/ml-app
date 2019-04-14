import pandas as pd
import lightgbm as lgb
from typing import List
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


import data.titanic.config as config


def run_lgb(train_X, train_y, val_X, val_y):
    params = {"objective": "regression", "metric": "mae", 'n_estimators': 20, 'early_stopping_rounds': 200,
              "num_leaves": 100, "learning_rate": 1, "bagging_fraction": 0.7,
              "bagging_seed": 0, "num_threads": 4, "colsample_bytree": 0.7
              }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)

    # pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)

    return model


def one_hot(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    for column in columns:
        df = pd.concat([df.drop(column, axis=1), pd.get_dummies(df[column], prefix=column + '-')], axis=1)
    return df


def correct(df: pd.DataFrame):
    # Age,  Embarked, Fare 三个字段有缺失, 需要填充

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)


def create(df: pd.DataFrame):
    # 构建家族大小

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0

    # 从名字获取头衔

    df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df.drop(['Name'], axis=1, inplace=True)

    # 对于小众, 杂乱的头衔, 统一标记为 'Misc'. 通过魔术数 10 来断定, 如果频数少于10, 替换
    title_names = (df['Title'].value_counts() < 10)
    df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] else x)

    # http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

    # Fare(费用)是连续值, 将其离散化

    df['FareBin'] = pd.qcut(df['Fare'], 4)

    # Age(年龄)是连续值, 将其离散化

    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)

    # cut 和 qcut 是两种不同的分桶方法


def convert(df: pd.DataFrame):
    label = LabelEncoder()
    df['AgeCode'] = label.fit_transform(train['AgeBin'])
    df['FareCode'] = label.fit_transform(train['FareBin'])

    for column in ['AgeCode', 'FareCode', 'Sex', 'Title', 'Embarked']:
        # df = df.append()
        df_dummy = pd.get_dummies(df[column], prefix=column)
        df[df_dummy.columns] = df_dummy
        # df.drop(column, axis=1)


# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

if __name__ == '__main__':
    train = pd.read_csv(config.file_train_csv)
    test = pd.read_csv(config.file_test_csv)

    # train_index = round(int(x_train.shape[0] * 0.8))
    # dev_X = x_train[:train_index]
    # val_X = x_train[train_index:]
    # dev_y = y_train[:train_index]
    # val_y = y_train[train_index:]

    # pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)

    # Cabin 缺失很多, 应该去掉
    correct(train)
    create(train)
    convert(train)

    print(train.info())

    # 检查各个变量与存活的相关性
    # group by 变量, 求 Survived 的均值就好
    for x in train:
        if train[x].dtype != 'float64' and x != 'Survived':
            print('Survival Correlation by:', x)
            print(train[[x, 'Survived']].groupby(x, as_index=False).mean())
            # using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
            print(pd.crosstab(train[x], train['Survived']))
            print('-' * 10, '\n')

    target_col = 'Survived'
    feature_cols = [
        'IsAlone',
        'AgeCode_0', 'AgeCode_1', 'AgeCode_2', 'AgeCode_3', 'AgeCode_4',
        'FareCode_0', 'FareCode_1', 'FareCode_2', 'FareCode_3',
        'Sex_female', 'Sex_male',
        'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Misc',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ]

    x_train = train[feature_cols]
    y_train = train[target_col]

    # correct(test)
    # create(test)
    # convert(test)
    #
    # x_test = test[feature_cols]

    train_index = round(int(x_train.shape[0] * 0.8))
    dev_X = x_train[:train_index]
    val_X = x_train[train_index:]
    dev_y = y_train[:train_index]
    val_y = y_train[train_index:]

    print(dev_X.shape)
    print(val_X.shape)
    print(dev_y.shape)
    print(val_y.shape)
    print(dev_X.info())

    # run_lgb(dev_X, dev_y, val_X, val_y)
    xgbc = XGBClassifier(max_depth=2, learning_rate=1, n_estimators=4, slient=False, objective="binary:logistic")

    xgbc.fit(dev_X, dev_y, verbose=True)

    y_train_pred = xgbc.predict(dev_X)
    y_train_pred = [round(x) for x in y_train_pred]
    train_score = accuracy_score(dev_y, y_train_pred)
    print("Train Accuracy: %.2f%%" % (train_score * 100))

    y_val_pred = xgbc.predict(val_X)
    y_val_pred = [round(x) for x in y_val_pred]
    cv_score = accuracy_score(val_y, y_val_pred)
    print("CV Accuracy: %.2f%%" % (cv_score * 100))





