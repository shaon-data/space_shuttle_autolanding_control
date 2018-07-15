# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,scale
from sklearn.linear_model import LinearRegression
from sklearn import neighbors,svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn import tree

data = pd.read_csv('data/shuttle-landing-control.csv',names=['auto_control','stability','error','sign','wind','magnitude','visibility'])
## |---------- Data Set Properties ------------|
#  |----- Value Map: 2 = True / 1 = False -----|
#  |------ Missing value: 0 -------------------|



## replacing missing values '*' with 0
data = data.replace('*',0)
## Always be aware of data type of a column, it can create error or unchange value when condition applied or other proccessing task applied.
## Converting data types into homogeneus element
data=data.astype('int')

features = data.ix[:,1:]
## Scalling has no effect on this data,
#Assuming that it is most of few categorical data
#features = StandardScaler().fit_transform(features)
target = data[data.columns[0]]

accuracies = []
models = []

clf =  LinearRegression()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))

clf = NearestCentroid()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))


clf = neighbors.KNeighborsClassifier()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))


#SVM
## 'precomputed' kernel function does not work, find the reason behind it, may be there is somme shape problem of datas
for kernel_function in ['rbf','linear', 'poly', 'rbf', 'sigmoid']:
    clf = svm.SVC(kernel = kernel_function)
    clf.fit(features,target)
    accuracies.append(clf.score(features,target))
    models.append(str(clf))



# Linear SVC
clf = svm.LinearSVC()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))

# Regression SVR
clf = svm.SVR()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))


# naive_bayes
clf =  GaussianNB()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))

# naive_bayes
clf =  BernoulliNB()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))

# naive_bayes
clf =  MultinomialNB()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))

# Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(features,target)
accuracies.append(clf.score(features,target))
models.append(str(clf))


for c,m in sorted(zip(accuracies,models),reverse=True):
    print(m)
    print("Accuracy: %s \n"%c)






#print("Accuracy:%s\n"%clf.score(features,target))


## Accuracy
## -- NearestCentroid           0.8
## -- KNeighborsClassifier(KNN) 0.7333333333333333
## -- SVM                       0.8
