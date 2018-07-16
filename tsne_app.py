# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import sys
sys.path.insert(0, "D:\work\codes\Ripositories\Data Science\My_Lib\EDA")

from EDA  import *

data = pd.read_csv('data/shuttle-landing-control.csv',names=['auto_control','stability','error','sign','wind','magnitude','visibility'])

## replacing missing values '*' with 0
data = data.replace('*',0)

## |---------- Data Set Properties ------------|
#  |----- Catagorical Value Map: 2 = True / 1 = False, 0 = Missing value -----|


## Always be aware of data type of a column, it can create error or unchange value when condition applied or other proccessing task applied.
## Converting data types into homogeneus element
data=data.astype('int')

scatter_matrix_graph_fit(data)

features = data.ix[:,1:]
target = data[data.columns[0]]
color = np.array(['g','r'])


target = target == 2
target = target.astype(int)


X = features
X_embedded = TSNE(n_components=3).fit_transform(X)
fig = plt.figure( )
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[:,0],X_embedded[:,1],X_embedded[:,2],c=color[target])

ax.scatter(X_embedded[:,0],X_embedded[:,1],c=color[target])
plt.show()

xS = 1
for yI in data.columns:
    yS = 1
    for xI in data.columns:
        yS = 1

        pearson_r(data[)
        yS+=1
    xS+=1


