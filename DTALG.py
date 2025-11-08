# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

def process(path):
	data=pd.read_csv(path)
	print("data.columns=",data.columns)
	X=data.iloc[1:data.shape[0],0:14]
	y=data.iloc[1:data.shape[0],14:15]
	#spliting the dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state =0 )
	

	#fitting logistic regression 
	classifier = DecisionTreeRegressor(random_state = 0)
	classifier.fit(X_train, y_train)

	#predicting the tests set result
	y_pred = classifier.predict(X_test)
	print("ypred",y_pred)
	
	
	result2=open("results/resultDT.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR Decision Tree IS %f "  % mse)
	print("MAE VALUE FOR Decision Tree IS %f "  % mae)
	print("R-SQUARED VALUE FOR Decision Tree IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR Decision Tree IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred.round())
	print ("ACCURACY VALUE Decision Tree IS %f" % ((ac)))
	print("---------------------------------------------------------")
	

	result2=open('results/DTMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str((ac)) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/DTMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('Decision Tree Metrics Value')
	fig.savefig('./static/results/DecissionTreeMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

process("./Dataset/dataset.csv")
