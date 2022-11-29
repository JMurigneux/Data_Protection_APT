import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,confusion_matrix,precision_score,recall_score,accuracy_score,f1_score
from sklearn.ensemble import GradientBoostingClassifier

import joblib

from xgboost import XGBClassifier,plot_importance,plot_tree

from math import sqrt
from statistics import mean

import tracemalloc
import time

label_encoder= preprocessing.LabelEncoder()

models = {
    "XGBoost" : XGBClassifier(),
    "KNN" : KNeighborsClassifier(),
    "CART":DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "SVM" : SVC(),
    "MLP" : MLPClassifier(),
}

SIZES = [1000,10000, 100000, 1000000]


def pretraitment(dataset,cut_value=0.5):
    dataset = dataset.drop("tags",axis=1)

    #drop the columns that contains the less information
    na = dataset.isna().sum()/dataset.shape[0]
    unusefull = []
    for name,value in na.items():
        if value > cut_value:
            unusefull.append(name)
    
    dataset = dataset.drop(unusefull, axis=1)
    
    X = dataset.copy()#.sample(frac=1,random_state=42).reset_index()
    Y = dataset["label"]
    Y.to_frame()
    X = X.drop(columns = ['label','raw','time','msg']) #columns that can't be used

    col = list(X.select_dtypes(include=['O']))

    enc = OneHotEncoder(handle_unknown='ignore')
    transformed = pd.DataFrame(enc.fit_transform(X[col]).toarray())
    X = X.join(transformed)
    X = X.drop(col,axis=1)
    return X,Y

def evaluation(model,X_test,y_test):
    y_test_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("confusion matrix")
    print(conf_matrix)
    if len(conf_matrix)<2:
        print("only one class")
        return conf_matrix
    
    tp = conf_matrix[1][1]
    fp = conf_matrix[0][1]
    tn = conf_matrix[0][0]
    fn = conf_matrix[1][0]
    
    print("true positive: ",tp)
    print("false positive: ",fp)
    print("true negative: ",tn)
    print("false negative: ",fn)

    print("++++++++++++++++++++++++++++++++++++++++")

    precision= tp/(tp+fp)
    recall=tp/(tp+fn)
    tnr=tn/(tn+fp)
    accuracy=(tp+tn)/(tp+fp+tn+fn)

    print("Balanced data metrics:")
    print("precision:" ,precision)
    print("recall:" ,recall)
    print("tnr:" ,tnr)
    print("accuracy:" ,accuracy)

    print("++++++++++++++++++++++++++++++++++++++++")

    f1_score=2 * (precision * recall) / (precision + recall)
    balanced_accuracy=(recall+tnr)/2
    matthews=(tn*tp-fn*fp)/(sqrt((tp+fp))*sqrt((tp+fn))*sqrt((tn+fp))*sqrt((tn+fn)))

    print("Unbalanced data metrics:")
    print("f1_score:" ,f1_score)
    print("balanced_accuracy:" ,balanced_accuracy)
    print("matthews correlation coefficient:" ,matthews)

    return conf_matrix


def init_model(model_name, train_and_test_dataset,size=0,with_evaluation = True):
    X,Y = pretraitment(train_and_test_dataset)

    if not size:
        X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.3, shuffle = True,random_state=42)
    else:
        ratio=size/len(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=ratio, shuffle = True,random_state=42)
    
    label_encoder.fit(Y)

    y_train_transformed = label_encoder.fit_transform(y_train)

    model = models[model_name]
    model.fit(X_train, y_train_transformed)

    if with_evaluation:
        evaluation(model, X_test=X_test, y_test=y_test)

    return model

def cat_pretraitment(dataset):
    dataset = dataset.drop("tags",axis=1)

    X = dataset.copy()#.sample(frac=1,random_state=42).reset_index()
    Y = dataset["label"]
    Y.to_frame()
    X = X.drop(columns = ['label','raw'])

    col = list(X.select_dtypes(include=['O']).columns)
    X[col]=X[col].astype('category')
    return X,Y


def cat_init_model(train_and_test_dataset,size=0, with_evaluation = True): #can only use xgboost
    X,Y = cat_pretraitment(train_and_test_dataset)

    if not size:
        X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.3, shuffle = True,random_state=42)
    else:
        ratio=size/len(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=ratio, shuffle = True,random_state=42)
        
    model = XGBClassifier(tree_method='hist',enable_categorical=True)
    model.fit(X_train, y_train)
    if with_evaluation:
        evaluation(model, X_test=X_test, y_test=y_test)

    return model

def cat_consumption_mesure(train_and_test_dataset,size):

    tracemalloc.start()
    start = time.time()

    model = cat_init_model(train_and_test_dataset,size=size, with_evaluation=True)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage peak was {peak / 10**6}MB")
    tracemalloc.stop()
    end = time.time()
    elapsed = end - start
    print(f"time elapsed : {int(elapsed)} sec")
    return

def is_malware(model, malware_features):
    labels = model.predict(malware_features)
    proba = model.predict_proba(malware_features)
    probability = []
    for i in range (len(labels)):
        probability.append(proba[i][labels[i]])
    return ( labels, probability)

def consumption_mesure(model_name, train_and_test_dataset,size=0):

    tracemalloc.start()
    start = time.time()

    if not size:
        model = init_model(model_name, train_and_test_dataset, with_evaluation=False)
    else:
        if train_and_test_dataset[:size]["label"].nunique()!=1 or model_name!="SVM":
            model = init_model(model_name, train_and_test_dataset,size=size, with_evaluation=True)
        else:
            print("SVM cannot be used with only one class")

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage peak was {peak / 10**6}MB")
    tracemalloc.stop()
    end = time.time()
    elapsed = end - start
    print(f"time elapsed : {int(elapsed)} sec")
    return

def save_model(model,path):
    joblib.dump(model, path)
    return

def load_model(path):
    model = joblib.load(path)
    return model

def test():
    # df = pd.read_csv("dataset/MSCAD.csv")
    print("BENCHMARK START")
    df = pd.read_csv("dataset/cleaned.csv",low_memory=False)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print("Dataset loaded")
    
    for size in SIZES:
        print("\n\n###################################################")
        print(f"for {size} rows\n\n")
        print("###################################################")
        for key,_ in models.items():
            print(f"\n\n\nfor model {key}\n\n")
            model = consumption_mesure(key,df,size)

        print(f"\n\n\nfor xgboost categorical\n\n")
        model=cat_consumption_mesure(df,size)
    # for key,_ in models.items():
    #     print(f"\n\n\nfor model {key}\n\n")
    #     for size in SIZES:
    #         print(f"for {size} rows\n\n")
    #         model = consumption_mesure(key,df,size)

test()
# print("start")
# df = pd.read_csv("dataset/cleaned.csv",low_memory=False)
# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# print("data loaded, start training")
# consumption_mesure("XGBoost",df,10000)  