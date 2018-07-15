import numpy as np
import pandas as pd

data = pd.read_csv('shuttle-landing-control.csv',names=['auto_control','stability','error','sign','wind','magnitude','visibility'])

## replacing missing values '*' with 0
data = data.replace('*',0)

## |---------- Data Set Properties ------------|
#  |----- Catagorical Value Map: 2 = True / 1 = False, 0 = Missing value -----|


## Always be aware of data type of a column, it can create error or unchange value when condition applied or other proccessing task applied.
## Converting data types into homogeneus element
data=data.astype('int')

## Assuming standardization is not needed at all value is mapped into same type of catagory
## Cross validation is not needed because data is too low

print(data.dtypes)
print(data.describe())

data.loc[data['auto_control']==1,'auto_control'] = False
data.loc[data['auto_control']==2,'auto_control'] = True

data.loc[data['visibility']==1,'visibility'] = False
data.loc[data['visibility']==2,'visibility'] = True



data.loc[data['sign']==1,'sign'] = '-'
data.loc[data['sign']==2,'sign'] = '+'

data.loc[data['wind']==1,'wind'] = 'tail'
data.loc[data['wind']==2,'wind'] = 'head'

data.loc[data['stability']==1,'stability'] = 'stab'
data.loc[data['stability']==2,'stability'] = 'xstab'

print(data)




