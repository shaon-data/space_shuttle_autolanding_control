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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# create some fake data
x = data['visibility']
y = data['wind']
# here are the x,y and respective z values
X, Y = np.meshgrid(x, y)
Z = np.sinc(np.sqrt(X*X+Y*Y))
# this is the value to use for the color
V = np.sin(Y)

# create the figure, add a 3d axis, set the viewing angle
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45,60)

# here we create the surface plot, but pass V through a colormap
# to create a different color for each patch
ax.plot_surface(X, Y, Z, facecolors=cm.Oranges(V))
ax.set_xlabel('visibility')
ax.set_ylabel('wind')
plt.title('Space Shuttle Auto-Landing Control')
plt.show()


