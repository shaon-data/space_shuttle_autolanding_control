# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,scale
from sklearn.linear_model import LinearRegression
from sklearn import neighbors,svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn import tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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
target = data.ix[:,0]


accuracies = []
models = []

KMeans
classifiers = [QuadraticDiscriminantAnalysis(),MLPClassifier(alpha=1),AdaBoostClassifier(),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),GaussianProcessClassifier(1.0 * RBF(1.0)),tree.DecisionTreeClassifier(),LinearRegression(),NearestCentroid(),neighbors.KNeighborsClassifier(),svm.LinearSVC()] + [svm.SVC(kernel = kernel_function) for kernel_function in ['rbf','linear', 'poly', 'rbf', 'sigmoid','linear'] ] + [svm.SVC(gamma=2, C=1, kernel = kernel_function) for kernel_function in ['rbf','linear', 'poly', 'rbf', 'sigmoid','linear'] ] + [svm.SVR(),GaussianNB(),BernoulliNB(),MultinomialNB()]
model_names = ['1']*len(classifiers)
model_names[23-1] = 'Regression SVR'
model_names[23:27] = ['naive_bayes']*3
print(model_names)
print(model_names.index("Regression SVR"))

###SVM > 'precomputed' kernel function does not work, find the reason behind it, may be there is somme shape problem of datas

## Running all the model
for class_model in classifiers:
    clf = class_model
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
