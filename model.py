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

from xgboost import XGBClassifier,plot_importance,plot_tree

from math import sqrt
from statistics import mean

import tracemalloc
import time
import psutil

label_encoder= preprocessing.LabelEncoder()

models = {
    "XGBoost" : XGBClassifier(),
    "KNN" : KNeighborsClassifier(),
    "CART":DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "SVM" : SVC(),
    "MLP" : MLPClassifier(),
}

# SIZES = [100, 10^3, 10^4, 10^5, 10^6, 10^7]
SIZES = [10^4, 10^5, 10^6,]


def pretraitment(dataset):
    X = dataset.copy()
    Y = dataset["Label"]
    Y.to_frame()
    X = X.drop(columns = ['Label'])
    col = list(dataset.select_dtypes(include=['O']))
    col = col[:-1]
    enc = OneHotEncoder(handle_unknown='ignore')
    transformed = pd.DataFrame(enc.fit_transform(X[col]).toarray())
    X = X.join(transformed)
    X = X.drop(col,axis=1)
    return X,Y

def evaluation(model,X_test,y_test):
    y_predictions_transformed = model.predict(X_test)

    y_test_t = label_encoder.transform(y_test)
    conf_matrix = confusion_matrix(y_test_t, y_predictions_transformed)
    sn.heatmap(conf_matrix)

    print("++++++++++++++++++++++++++++++++++++++++")
    n=len(conf_matrix)
    columns = conf_matrix.sum(axis=0)
    for i in range(0,n):
        tp = conf_matrix[i][i]
        fn = conf_matrix[i].sum() - tp
        fp = columns[i] - tp
        tn = conf_matrix.sum() - (tp + fn + fp)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        TNR = tn/(tn+fp)
        sensitivy = tp/(tp+fn)
        specifity = tn/(fp+tn)

        print(f"Class {i}\n")
        print("\n\tBalanced data :\n")
        print(f"Precision : {precision}\n")
        print(f"Recall : {recall}\n")
        print(f"True Negative Rate : {TNR}\n")
        print(f"Accuracy : {(recall+TNR)/2}\n")
        print("\n\tUnbalanced data :\n")
        print(f"F1-score : {2*precision*recall/(precision+recall)}\n")
        print(f"Balanced accuracy : {(sensitivy+specifity)/2}\n")
        print(f"Matthews Correlation Coefficient : {(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))}\n\n")
    
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)
    print("++++++++++++++++++++++++++++++++++++++++")
    print(f"Total precision : {precision_score(y_test_t,y_predictions_transformed,average='micro')}\n")
    print(f"Total recall : {recall_score(y_test_t,y_predictions_transformed,average='micro')}\n")
    print(f"Total True Negative Rate : {mean(TN/(TN+FP)) }\n")
    print(f"Total accuracy : {accuracy_score(y_test_t,y_predictions_transformed,)}\n")
    print(f"Total F1-score : {f1_score(y_test_t,y_predictions_transformed,average='micro')}\n")
    print(f"Total balanced accuracy : {balanced_accuracy_score(y_test_t,y_predictions_transformed)}\n")
    print(f"Total Matthews Correlation Coefficient : {matthews_corrcoef(y_test_t,y_predictions_transformed)}\n\n")

    return


def init_model(model_name, train_and_test_dataset, with_evaluation = True):
    X,Y = pretraitment(train_and_test_dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.3, shuffle = True)
    
    label_encoder.fit(Y)

    y_train_transformed = label_encoder.fit_transform(y_train)

    model = models[model_name]
    model.fit(X_train, y_train_transformed)

    if with_evaluation:
        evaluation(model, X_test=X_test, y_test=y_test)

    return model

def is_malware(self, model, malware_features):
    Boolean,probability = False, 0
    return (Boolean, probability)

def consumption_mesure(model_name, train_and_test_dataset,size=0):

    tracemalloc.start()
    start = time.time()

    if not size:
        model = init_model(model_name, train_and_test_dataset, with_evaluation=False)
    else:
        model = init_model(model_name, train_and_test_dataset[:size], with_evaluation=True)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage peak was {peak / 10**6}MB")
    tracemalloc.stop()
    end = time.time()
    elapsed = end - start
    print(elapsed)
    print(psutil.cpu_times_percent(interval=1))
    return


def test():
    df = pd.read_csv("dataset/MSCAD.csv")
    for size in SIZES:
        model = consumption_mesure("XGBoost",df,size)

test()
#df = pd.read_csv("dataset/MSCAD.csv")
#init_model("XGBoost",df, with_evaluation=True)
