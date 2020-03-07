import pickle
import xgboost as xgb
import numpy as np
##read the dataset
X_train=np.loadtxt("TrainData/TrainData.csv")
y_train=np.loadtxt("TrainLabels.csv")
#train the model on the entire dataset
clf = xgb.XGBClassifier(n_estimators = 150,max_depth = 100,n_jobs = -1)
clf.fit(X_train,y_train)
##save the model into the system
pickle.dump(clf, open('xgboost.pickle', 'wb'))