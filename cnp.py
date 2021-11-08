import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import GenericUnivariateSelect


def preprocessing():
    df = pd.read_csv("CNPC_1401-1509_DI_v1_1_2016-03-01.csv",
                     na_values=["NA", "NAN", "Nan", "na", "Missing", "missing", "n/a", "nan"])
    # We fill all the null and NaN and missing values with zero.

    df["LoE_DI"] = df["LoE_DI"].fillna(0)
    df["LoE_DI"].replace({"Master's Degree (or equivalent)": 1, "Completed 4-year college degree": 2,
                          "Some college, but have not finished a degree": 3, "Some graduate school": 4,
                          "Ph.D., J.D., or M.D. (or equivalent)": 5, "Completed 2-year college degree": 6,
                          "High School or College Preparatory School": 7, 'None of these': 8}, inplace=True)
    df = df.fillna(0)
    df["learner_type"] = df["learner_type"].fillna(0)
    df["learner_type"].replace(
        {"Drop-in": "Observer", "Passive participant": "Passive", "Active participant": "Active"}, inplace=True)
    lst = df["learner_type"].values.tolist()
    # We now normalize them and get those values into a new list
    le = LabelEncoder()
    le.fit(lst)
    list(le.classes_)
    lt = le.transform(lst)
    # we drop the existing column and use the new processed list column as learner_type
    df = df.drop(columns=['learner_type'])
    learner_type = lt.tolist()
    df.insert(3, "learner_type", learner_type)
    # Similarly we normalize expected_hours_week and do fit transform and add this in place of existing one
    lst1 = df["expected_hours_week"].values.tolist()
    le.fit(lst1)
    list(le.classes_)
    lt1 = le.transform(lst1)
    df = df.drop(columns=['expected_hours_week'])
    expected_hours_week = lt1.tolist()
    df.insert(4, "expected_hours_week", expected_hours_week)
    # For fitting the model aand for best accuracy we convert selected attributes to a single type, into int
    df["nevents"] = df["nevents"].astype(int)
    df["nforum_posts"] = df["nforum_posts"].astype(int)
    df["ndays_act"] = df["ndays_act"].astype(int)
    df["completed_%"] = df["completed_%"].astype(int)
    return df


def feature_selection(df):
    # Here we are checking which is the best feature for us using Correlaation with heatmap
    correlationmatrix = df.corr()
    selected_features = correlationmatrix.index
    plt.figure(figsize=(10, 10))
    # plot heat map
    heatmap = sns.heatmap(df[selected_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()

def generic_univariate_select(df):
    df_gus = df.drop(
        ['discipline', 'primary_reason', 'start_time_DI', 'course_start', 'course_end', 'last_event_DI', 'age_DI',
         'final_cc_cname_DI', 'gender'], axis=1)
    X = df_gus.iloc[:, :-1]
    y = df_gus.iloc[:, -1]
    trans = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=50)
    chars_X_trans = trans.fit_transform(X, y)
    print(chars_X_trans)
    return X,chars_X_trans





def Random_forest(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    rfc = RandomForestRegressor().fit(X_train, np.ravel(y_train,order='C'))
    pred_rfc = rfc.predict(X_test)

    errors = abs(np.mean(pred_rfc) - np.mean(y_test))
    mean_absolute_percent_error = 100 * (errors / np.mean(y_test))
    accuracy = 100 - mean_absolute_percent_error
    print("-----------RANDOM__FOREST--------------------")
    print('Accuracy:', round(accuracy, 2), '%.')
    mse = mean_squared_error(y_test, pred_rfc)
    rmse = np.sqrt(mse)
    print("RMSE: ",rmse)

def Bayesian_ridge(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    clf = BayesianRidge()
    clf.fit(X_train, np.ravel(y_train,order='C'))
    y_pred_gnb = clf.predict(X_test)


    error = abs(np.mean(y_pred_gnb) - np.mean(y_test))
    mean_absolute_percent_error = 100 * (error / np.mean(y_test))
    accuracy = 100 - mean_absolute_percent_error
    print("----------BAYESIAN_RIDGE-------------")
    print('Accuracy:', round(accuracy, 2), '%.')
    gnbmse = mean_squared_error(y_test, y_pred_gnb)
    gnbrmse = np.sqrt(gnbmse)
    print("RMSE: ", gnbrmse)

def Linear_regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, np.ravel(y_train,order='C'))
    y_pred_lr = regr.predict(X_test)
    print("----------LINEAR_REGRESSION-----------------")
    # The coefficients for Linear-regression
    print('Coefficients for Linear-regression: \n', regr.coef_)
    # The mean squared error for Linear-regression
    print('Mean squared error for Linear-regression:', mean_squared_error(y_test, y_pred_lr))
    # The coefficient of determination for Linear-regression
    print('Coefficient of determination for Linear-regression:', r2_score(y_test, y_pred_lr))


    error = abs(np.mean(y_pred_lr) - np.mean(y_test))
    mean_absolute_percent_error = 100 * (error / np.mean(y_test))
    accuracy = 100 - mean_absolute_percent_error
    print('Accuracy:', round(accuracy, 2), '%.')
    lrmse = mean_squared_error(y_test, y_pred_lr)
    lrrmse = np.sqrt(lrmse)
    print("RMSE: ",lrrmse)



if __name__ == '__main__':
    # First Pre-process data
    df=preprocessing()
    # Performing feature Selection methods so as to select features which are important for grade prediction
    feature_selection(df)
    X,chars_X_trans = generic_univariate_select(df)
    print("We started with {0} pixels but retained only {1} of them!".format(X.shape[1], chars_X_trans.shape[1]))
    # Dropping columns which we found as not important with respect to the heat map
    df = df.drop(['registered', 'primary_reason', 'start_time_DI', 'last_event_DI', 'course_reqs',
                   'final_cc_cname_DI', 'age_DI', 'gender', 'course_start', 'course_end',
                  'course_length', 'course_id_DI', 'discipline', 'userid_DI'], axis=1)
    # Selected important features and how they vary with respect to grade after dropping some features
    feature_selection(df)
    df.info
    # Apply Algorithms
    X = df[['viewed', 'grade_reqs', 'learner_type', 'nforum_posts', 'ndays_act','ncontent','LoE_DI', 'explored','nevents',
            'expected_hours_week']]
    y = df[['grade']]
    # Apply Random-Forest
    Random_forest(X,y)
    # Apply Bayesian-Ridge
    Bayesian_ridge(X,y)
    # Apply Linear-Regression
    Linear_regression(X,y)