#Artificial nural network
#Data preprocessing
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#encoding categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
X[:,1]=labelencoder_x1.fit_transform(X[:,1])

labelencoder_x2=LabelEncoder()
X[:,2]=labelencoder_x2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#splitting dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier=Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

#Adding the second hidden layer 
classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))

# Adding output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

#compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 #fit Ann to traning set
classifier.fit(x_train,y_train,batch_size=10,epochs=100)

#prediction the test set result
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Use ANN model to predict if the customer with the following informations will leave the bank: 
#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000
new_pred=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>0.5)





















