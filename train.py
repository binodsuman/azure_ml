# -*- coding: utf-8 -*-
"""
@author: agarw
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc
import pandas as pd
import joblib
import argparse

from azureml.core import Run

run = Run.get_context()

parser=argparse.ArgumentParser()

parser.add_argument('--min_samples_leaf',type=int)
parser.add_argument('--min_samples_split',type=int)

args=parser.parse_args()


data=pd.read_csv("./diabetes.csv")
print(data.columns)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

trainx,testx,trainy,testy=train_test_split(X, y, random_state=101, test_size=0.2)

lg=RandomForestClassifier(min_samples_leaf= args.min_samples_leaf, min_samples_split=args.min_samples_split)
lg.fit(trainx,trainy)
print(lg.get_params())
run.log("Parameter",lg.get_params())
y_pred=lg.predict(testx)

classification_report(testy,y_pred)

confusion_matrix(testy,y_pred)
acc=accuracy_score(testy,y_pred)
run.log("Accuracy",acc)

print(accuracy_score(testy,y_pred))
joblib.dump(lg, "diabeticmodel.pkl")

