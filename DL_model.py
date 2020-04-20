#Step 1 — Data Pre-processing
import pandas as pd
import numpy as np
df = pd.read_csv('HR_comma_sep.csv')
feats = ['sales','salary']
df_final = pd.get_dummies(df,columns=feats,drop_first=True)#converting the feats from categorical to numerical variables
#Step 2 — Separating Your Training and Testing Datasets
from sklearn.model_selection import train_test_split
X = df_final.drop(['left'],axis=1).values#define the variables
y = df_final['left'].values#define the target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#split data into 70% for training and 30% for testing
#Step 3 — Transforming the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Step 4 — Building the Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
from keras.layers import Dropout
classifier.add(Dense(9, kernel_initializer = "uniform",activation = "relu", input_dim=18))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 5)

#Step 5 — Running Predictions on the Test Set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
#Step 6 — Checking the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix :' , cm)
score=classifier.evaluate(X_test,y_test,verbose=0)
print(classifier.metrics_names[1],score[1]*100)
new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))
new_pred=(new_pred>0.5)
print("new_pred :",new_pred)
