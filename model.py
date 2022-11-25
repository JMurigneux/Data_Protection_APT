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
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,precision_recall_curve,confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier,plot_importance,plot_tree

import tracemalloc

label_encoder= preprocessing.LabelEncoder()

models = {
    "XGBoost" : XGBClassifier(),
    "KNN" : KNeighborsClassifier(),
    "CART":DecisionTreeClassifier(),
    "Random Forrest" : RandomForestClassifier(),
    "SVM" : SVC(),
    "MLP" : MLPClassifier(),
}

# SIZES = [100, 10^3, 10^4, 10^5, 10^6, 10^7]
SIZES = [10^4, 10^5, 10^6, 10^7]


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

    print("The precision, the recall, and the accuracy can be found in the following classification report :")
    print(classification_report(y_test_t, y_predictions_transformed))
    print("++++++++++++++++++++++++++++++++++++++++")
    print("The balanced accuracy : " + str(balanced_accuracy_score(y_test_t, y_predictions_transformed)))
    print("The Matthews Correlation Coefficient : " + str(matthews_corrcoef(y_test_t,y_predictions_transformed)))
    return

def init_model(model_name, train_and_test_dataset, with_evaluation = True):
    X,Y = pretraitment(train_and_test_dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,train_size=0.3, shuffle = True)
    
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
    if not size:
        model = init_model(model_name, train_and_test_dataset, with_evaluation=False)
    else:
        model = init_model(model_name, train_and_test_dataset.sample(frac=1)[:size], with_evaluation=True)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage peak was {peak / 10**6}MB")
    tracemalloc.stop()
    return


def test():
    df = pd.read_csv("dataset/MSCAD.csv")
    for size in SIZES:
        model = consumption_mesure("XGBoost",df,size)

test()
