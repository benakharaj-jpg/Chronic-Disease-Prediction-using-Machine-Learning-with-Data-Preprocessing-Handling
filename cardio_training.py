import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop,Nadam,Adadelta,Adam
from tensorflow.keras.layers import BatchNormalization,LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import seaborn as sns
import scipy.stats as stats
import sklearn
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

df = mydata = pd.read_csv("Data/cardio_train.csv", sep=";")

df.drop('id', inplace=True, axis=1)
df.head()
dfcol = df.columns



from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
dfscale=scaler.fit_transform(df)
dfscale2=pd.DataFrame(dfscale, columns=dfcol)
dfscale2.head()

# x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# x_train = normalize(x_train)
# x_test = normalize(x_test)
# x = normalize(x)


xdf=dfscale2.iloc[:,0:11]
#xdf["gender"]=np.where(xdf["gender"]==1,"0","1") #Cambiar el 2 por 1, el 1 por 0 (por orden)
#Aca vendria un posible drop de variables xdf=xdf.drop(["gender","gluc"], axis=1)
ydf=dfscale2.iloc[:,-1]

x_training, x_testing, y_training, y_testing = train_test_split(xdf, ydf, test_size = 0.2, random_state=123, stratify=ydf)


ran = RandomForestClassifier(n_estimators=100)

ran2 = ran.fit(x_training,y_training)




filename = 'fcardio.sav'
pickle.dump(ran2, open(filename, 'wb'))