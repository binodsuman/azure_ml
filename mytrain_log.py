from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc
import pandas as pd
import joblib
import argparse

from azureml.core import Run

run = Run.get_context()

data=pd.read_csv("./diabetes.csv")
print(data.columns)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

trainx,testx,trainy,testy=train_test_split(X, y, random_state=101, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(trainx, trainy)
run.log("Parameter",rf.get_params())
print("Accuracy on training set: {:.3f}".format(rf.score(trainx, trainy)))
print("Accuracy on test set: {:.3f}".format(rf.score(testx, testy)))

y_pred=rf.predict(testx)
classification_report(testy,y_pred)
confusion_matrix(testy,y_pred)

acc=accuracy_score(testy,y_pred)
run.log("Accuracy",acc)

print(accuracy_score(testy,y_pred))
joblib.dump(rf, "diabeticmodel.pkl")