
# coding: utf-8

# In[1]:


import timeit
import time
import sys
# !{sys.executable} -m pip install lightgbm
from sklearn.preprocessing import MinMaxScaler, Imputer,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from contextlib import contextmanager
from sklearn.decomposition import PCA
#from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
import gc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import numpy as np
from fancyimpute import KNN
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import missingno as mn

import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()


# from plotly import tools
# import plotly.tools as tls
# import squarify
# from mpl_toolkits.basemap import Basemap
# from numpy import array
# from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()


# # Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# # Print all rows and columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

pd.set_option('display.max_rows', 500)
print(np.__version__)
pd.options.display.max_columns = 500



# In[2]:





# In[3]:


def var_memory():
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    # Get a sorted list of the objects and their sizes
    var=sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_')], key=lambda x: x[1], reverse=True)
    return var
def missing_value_summary(df):
    total= df.isnull().sum().sort_values(ascending = False)
    percentage= (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_df_data=pd.concat([total,percentage], axis=1, keys=['Total','Percent'])
    del df
    gc.collect()
    return missing_df_data

def missing_value_vis(df):
    mv_vis=mn.matrix(df)
    del df
    gc.collect()
    return mv_vis

def imputation(df, columns):
    df= df.drop(columns, axis=1)
    occupation_mode= df["OCCUPATION_TYPE"].value_counts()[0]
    #We impute with most common occupation for now
    df["OCCUPATION_TYPE"]=df["OCCUPATION_TYPE"].fillna(value=occupation_mode)

    #For people who do not own cars, we impute OWN_CAR_AGE with 0
    m= (df["OWN_CAR_AGE"].isnull())&(df["FLAG_OWN_CAR"]=="N")
    df.loc[m,"OWN_CAR_AGE"]=df.loc[m,"OWN_CAR_AGE"].fillna(0)
    own_car_median= df["OWN_CAR_AGE"].median()
    df["OWN_CAR_AGE"].fillna(own_car_median, inplace=True)

    # imputing with median
    df[["EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_QRT", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
       "OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE", "EXT_SOURCE_2", "AMT_GOODS_PRICE",
       "AMT_ANNUITY","CNT_FAM_MEMBERS","DAYS_LAST_PHONE_CHANGE"]] = df[["EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_QRT", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
       "OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE", "EXT_SOURCE_2", "AMT_GOODS_PRICE",
       "AMT_ANNUITY","CNT_FAM_MEMBERS","DAYS_LAST_PHONE_CHANGE"]].apply(lambda x : x.fillna(x.median()), axis=0)
    name_type_mode=df["NAME_TYPE_SUITE"].value_counts()[0]
    df["NAME_TYPE_SUITE"].fillna(name_type_mode, inplace=True)

    gc.collect()

    #imputing EXT_SOURCE_1 and EXT_SOURCE_3
    return df


def delete_columns(df,columns):
    return df.drop(columns, axis=1)

def impute_median(df, columns):
    medians= df[columns].median()
    for col in columns:
        df[[col]].fillna(medians[col], inplace=True)
    #df[columns]= df[columns].apply( lambda x: x.fillna(x.median()), axis=1 )
    gc.collect()
    return df

def feature_engineering_bureau_balance(df):
    cat_aggregations={
        'STATUS_0':["sum"],
        'STATUS_1':["sum"],
        'STATUS_2':["sum"],
        'STATUS_3':["sum"],
        'STATUS_4':["sum"],
        'STATUS_5':["sum"],
        'STATUS_C':["sum"],
        'STATUS_X':["sum"]
    }
    aggdf= df.groupby("SK_ID_BUREAU").agg(cat_aggregations)
    aggdf.columns=["BB"+"_"+e[0]+"_"+e[1] for e in aggdf.columns]
    del df
    gc.collect()
    return aggdf
def feature_engineering_bureau(df):
    num_aggregations={
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE':['sum', 'mean','max'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_ENDDATE_FACT':['mean','max', 'min'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean','max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        #'MONTHS_BALANCE_MIN': ['min'],
        #'MONTHS_BALANCE_MAX': ['max'],
        #'MONTHS_BALANCE_SIZE': ['mean', 'sum']

    }
    cat_aggregations={
        #We can also create good to bad loan ratios
        'CREDIT_ACTIVE_Bad debt':["sum","mean"],
        'CREDIT_ACTIVE_Closed':["sum","mean"],
        'CREDIT_ACTIVE_Sold':["sum","mean"],
        'CREDIT_CURRENCY_currency 2':["sum","mean"],
        'CREDIT_CURRENCY_currency 3':["sum","mean"],
        'CREDIT_CURRENCY_currency 4':["sum","mean"],
        'CREDIT_TYPE_Car loan':["sum","mean"],
        'CREDIT_TYPE_Cash loan (non-earmarked)':["sum","mean"],
        'CREDIT_TYPE_Consumer credit':["sum","mean"],
        'CREDIT_TYPE_Credit card':["sum","mean"],
        'CREDIT_TYPE_Interbank credit':["sum","mean"],
        'CREDIT_TYPE_Loan for business development':["sum","mean"],
        'CREDIT_TYPE_Loan for purchase of shares (margin lending)':["sum","mean"],
        'CREDIT_TYPE_Loan for the purchase of equipment':["sum","mean"],
        'CREDIT_TYPE_Loan for working capital replenishment':["sum","mean"],
        'CREDIT_TYPE_Microloan':["sum","mean"],
        'CREDIT_TYPE_Mobile operator loan':["sum","mean"],
        'CREDIT_TYPE_Mortgage':["sum","mean"],
        'CREDIT_TYPE_Real estate loan':["sum","mean"],
        'CREDIT_TYPE_Unknown type of loan':["sum","mean"],
        'BB_STATUS_0_sum':["sum","mean"],
        'BB_STATUS_1_sum':["sum","mean"],
        'BB_STATUS_2_sum':["sum","mean"],
        'BB_STATUS_3_sum':["sum","mean"],
        'BB_STATUS_4_sum':["sum","mean"],
        'BB_STATUS_5_sum':["sum","mean"],
        'BB_STATUS_C_sum':["sum","mean"],
        'BB_STATUS_X_sum':["sum","mean"]
    }
    aggdf=df.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    aggdf.columns = ["BUREAU_"+e[0]+"_"+e[1] for e in aggdf.columns]
    del df
    gc.collect()
    return aggdf

def one_hot_encoder(df, nan_as_category=True, drop_first=True):
    original_columns = list(df.columns)
    categorical_columns=[col for col in original_columns if df[col].dtype=='object']
    df=pd.get_dummies(data=df, columns=categorical_columns, dummy_na=nan_as_category, drop_first=drop_first)
    new_columns = [c for c in df.columns if c not in original_columns]
    gc.collect()
    return df, new_columns

def feature_engineering_previous(df, cat_cols):

    num_aggregations = {
        'AMT_ANNUITY': ['min','max','mean'],
        'AMT_APPLICATION': ['min','max','mean'],
        'AMT_CREDIT': ['min','max','mean'],
        'APP_CREDIT_PERC': ['min','max','mean'],
        'AMT_DOWN_PAYMENT': ['min','max','mean'],
        'AMT_GOODS_PRICE': ['min','max','mean'],
        'HOUR_APPR_PROCESS_START': ['min','max','mean'],
        'RATE_DOWN_PAYMENT': ['min','max','mean'],
        'DAYS_DECISION': ['min','max','mean'],
        'CNT_PAYMENT': ['min','sum']
    }
    cat_aggregations=dict()
    for cat in cat_cols:
        cat_aggregations[cat]= ['mean', 'sum']
    aggdf=df.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    aggdf.columns = ["PREV_"+e[0]+"_"+e[1] for e in aggdf.columns]
    del df
    gc.collect()
    return aggdf

def feature_engineering_POS_CASH_balance(df, cat_cols):
    num_aggregations = {
        'MONTHS_BALANCE':['max', 'min', 'size'],
        'SK_DPD':['max', 'min','mean'],
        'SK_DPD_DEF':['max', 'min','mean']
    }
    cat_aggregations={}
    for col in cat_cols:
        cat_aggregations[col]= ['sum','mean']
    aggdf= df.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    aggdf.columns = ["POS_"+e[0]+"_"+e[1] for e in aggdf.columns]
    del df
    gc.collect()
    return aggdf

def feature_engineering_installments(df):
    df['DAY_DIFF'] = df['DAYS_ENTRY_PAYMENT'] = df['DAYS_INSTALMENT']
    df['TIMELY'] = df['DAY_DIFF'].apply(lambda x: 1 if x>0 else 0)
    df['PAY_PERC'] = df['AMT_PAYMENT']/df['AMT_INSTALMENT']
    df['PAY_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    aggregations= {
        'TIMELY':['sum','mean'],
        'PAY_PERC':['max', 'mean', 'sum','var'],
        'PAY_DIFF':['max', 'mean', 'sum','var'],
        'AMT_PAYMENT':['max','mean','sum'],
        'AMT_INSTALMENT':['max', 'mean', 'sum'],
        'DAY_DIFF':['min','max', 'mean','sum']

    }
    aggdf=df.groupby('SK_ID_CURR').agg(aggregations)
    aggdf.columns= ["INS_"+e[0]+"_"+e[1] for e in aggdf.columns]
    del df
    gc.collect()
    return aggdf


def feature_engineering_credit(df,cv=5):
    aggdf=df.groupby('SK_ID_CURR').agg(['min','max', 'sum','var','mean'])
    aggdf.columns=["CC_"+e[0]+"_"+e[1] for e in aggdf.columns]
    aggdf["CC_COUNT"] = df.groupby("SK_ID_CURR").size()
    del df
    gc.collect()
    return aggdf

def preprocessing_imputer(df,n_train):
    #train= df.iloc[:n_train, :]
    train_X = df.iloc[:n_train, :].drop(['TARGET','SK_ID_CURR', 'index'],axis=1)
    train_y = df.iloc[:n_train, :].TARGET
    #test= df.iloc[n_train:,:]
    test_X=df.iloc[n_train:,:].drop(['TARGET','SK_ID_CURR', 'index'],axis=1)
    features=list(train_X.columns)
    imputer= Imputer(strategy='median')
    imputer.fit(train_X)
    train_X=imputer.transform(train_X)
    test_X=imputer.transform(test_X)
    del df
    gc.collect()
    return train_X, test_X, train_y, features

def preprocessing_scaler(train_X,test_X):
    scaler=StandardScaler()
    scaler.fit(train_X)
    train_X=scaler.transform(train_X)
    test_X=scaler.transform(test_X)
    gc.collect()
    return train_X, test_X

def logistic_regression(train_X, train_y, cv=5):
    lr= LogisticRegressionCV(class_weight='balanced',cv=cv)
    lr.fit(train_X,train_y)
    gc.collect()
    return lr

def random_forest(train_X, train_y,cv=5):
    rf= RandomForestClassifier(n_estimators=100, random_state=50, class_weight='balanced')
    rf.fit(train_X, train_y)
    return rf

def feature_importance(model, features):
    plt.figure(figsize=(12,8))
    res = pd.DataFrame({'feature':features, 'importance': model.feature_importances_})
    res=res.sort_values('importance', ascending= False)
    print(res.head(10))
    print("Features > 0.01: ", np.sum(res['importance']> 0.01))

    res.head(20).plot(x='feature', y='importance', kind='barh',
                     color='red', edgecolor='k', title='Feature Importances')
    return res

def main():
    print("dataset read begin")
    comb=pd.read_csv("dataset.csv")
    print("dataset read done")
    n_train=307511
    comb= comb.convert_objects(convert_numeric=True)
    print("imputing begin")
    train_X, test_X, train_y, features= preprocessing_imputer(comb, n_train)
    print("imputing done")
    del comb
    gc.collect()
    tr=np.isinf(train_X)
    te=np.isinf(test_X)
    train_X[tr]=1e10
    test_X[te]=1e10
    print("scaling begin")
    train_X, test_X=preprocessing_scaler(train_X, test_X)
    print("scaling done")
    print("rf begin")
    rfmodel= random_forest(train_X, train_y)
    print("rf begin")
    #results=feature_importance(rfmodel, features)
    predictions=rfmodel.predict_proba(test_X)[:,1]
    predictions[0:10]
    print("submission read begin")
    sample=pd.read_csv('sample_submission.csv')
    print("submission read begin")
    sample.TARGET=predictions
    sample.to_csv("Submission3.csv", index=False)
    print("lr begin")
    lrmodel=logistic_regression(train_X,train_y)
    print("lr done")
    predictions= lrmodel.predict_proba(test_X)[:,1]
    sample.TARGET=predictions
    sample.to_csv("Submission4.csv", index=False)



if __name__ == '__main__':
    main()
