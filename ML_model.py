#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

#Data analysis
dataset = pd.read_csv("HR_comma_sep.csv")
data = dataset.copy()

correlation = data.corr()
correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1,vmin=-1,square=True,annot=True,linewidths=.5,cmap="YlGnBu")
plt.show()
leave = data[(data['left'] == 1)]
stay = data[(data['left']==0)]
dept_name = data['sales'].unique()
name=['Sales','Accounting','HR','Technical','Support','Management','IT','Product Management','Marketing','RandD']
index = range(10)

plt.figure(1,figsize=(12,8))

plt.subplot(1,2,1)
leave['sales'].value_counts().plot(kind='bar')
plt.title('Employees who "LEAVE" the company by department')
plt.xticks(index,name)


plt.subplot(1,2,2)
stay['sales'].value_counts().plot(kind='bar',color='orange')
plt.title('Employees who "STAY" in the company by department')
plt.xticks(index,name)

leave_time_spent_count = leave['time_spend_company'].value_counts().sort_index()
stay_time_spent_count = stay['time_spend_company'].value_counts().sort_index()

plt.figure(1,figsize=(12,8))
plt.subplot(1,2,1)
leave_time_spent_count.plot(kind='bar',rot=0)
plt.title('Years spent in the \n company before LEAVING')
plt.xlabel('Years')
plt.ylabel('No. of employees')

plt.subplot(1,2,2)
stay_time_spent_count.plot(kind='bar',rot=0,color='green')
plt.title('Years spent in the \n company without leaving(STAYING)')
plt.xlabel('Years')
plt.ylabel('No. of employees')
plt.show()

#Prediction Analysis
feats = ['sales','salary']
df_final = pd.get_dummies(data,columns=feats,drop_first=True)
X = df_final.drop(['left'],axis=1).values
y = df_final['left'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Rf=RandomForestClassifier()
Rf.fit(X_train,y_train)
Lr=LogisticRegression()
Lr.fit(X_train,y_train)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
svc_linear=SVC()
svc_linear.fit(X_train,y_train)
print ("Random Forest Classifier accuracy :",Rf.score(X_test,y_test))
print ("Logistic Regression accuracy :",Lr.score(X_test,y_test))
print ("KNeighborsClassifier accuracy :",clf.score(X_test,y_test))
print ("SVC accuracy :",svc_linear.score(X_test,y_test))
new_pred = np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])
prediction=Rf.predict(new_pred)
print('RandomForest new_pred :', prediction)

