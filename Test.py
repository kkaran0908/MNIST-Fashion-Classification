##import all the required libraries
import pickle
import pandas as pd
#reading the test data
X_test = np.loadtxt('TestData.csv')
#load the pretrained model
model = pickle.load(open('xgboost.pickle', 'rb'))
prediction = model.predict(X_test)
np.savetxt("myPredictions.csv", prediction) 