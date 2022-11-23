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

class cmodel():
    le = preprocessing.LabelEncoder()

    models = {
        "XGBoost" : XGBClassifier(),
        "KNN" : KNeighborsClassifier(),
        "CART":DecisionTreeClassifier(),
        "Random Forrest" : RandomForestClassifier(),
        "SVM" : SVC(),
        "MLP" : MLPClassifier(),
    }


    def pretraitment(self, dataset):
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

    def init_model(self, model_name, train_and_test_dataset):
        X,Y = self.pretraitment(train_and_test_dataset)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y,train_size=0.3, shuffle = True)
        
        y_train_transformed = self.le.fit_transform(self.y_train)

        self.models[model_name].fit(self.X_train, y_train_transformed)

        return self.models[model_name]

    def evaluation(self,model_name):
        y_predictions_transformed = self.models[model_name].predict(self.X_test)

        y_test_t = self.le.transform(self.y_test)
        conf_matrix = confusion_matrix(y_test_t, y_predictions_transformed)
        sn.heatmap(conf_matrix)

        print("The precision, the recall, and the accuracy can be found in the following classification report :")
        print(classification_report(y_test_t, y_predictions_transformed))

        print("The balanced accuracy : " + str(balanced_accuracy_score(y_test_t, y_predictions_transformed)))
        print("The Matthews Correlation Coefficient : " + str(matthews_corrcoef(y_test_t,y_predictions_transformed)))
        return

    def is_malware(self, model, malware_features):
        Boolean,probability = False, 0
        return (Boolean, probability)


df = pd.read_csv("dataset/MSCAD.csv")
xgb_model = cmodel()
xgb_model.init_model("XGBoost",df)
xgb_model.evaluation("XGBoost")