from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm

from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
def process(path):
	data=pd.read_csv(path)
	
	X=data.iloc[1:data.shape[0],0:14]
	y=data.iloc[1:data.shape[0],14:15]
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
	
	model2=XGBClassifier(objective='multi:softprob')
	model2.fit(X_train, y_train)
	y_pred = model2.predict(X_test)
	print("predicted")
	print(y_pred)
	print(y_test)
	result2=open("results/resultXGB.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR XGBoost IS %f "  % mse)
	print("MAE VALUE FOR XGBoost IS %f "  % mae)
	print("R-SQUARED VALUE FOR XGBoost IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR XGBoost IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred)
	print ("ACCURACY VALUE XGBoost IS %f" % (ac))
	print("---------------------------------------------------------")
	

	result2=open('results/XGBMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/XGBMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('XGBoost Value')
	fig.savefig('./static/results/XGBMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

process("./Dataset/dataset.csv")
