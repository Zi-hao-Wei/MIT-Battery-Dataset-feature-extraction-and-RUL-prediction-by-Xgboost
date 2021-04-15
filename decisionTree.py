import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import tree 

import pickle
import matplotlib.pyplot as plt 
import math 
data=pd.read_csv(r"regularized.csv",low_memory=False)

drop=["Unnamed: 0","lifetime","lifePercent"]
train_data=data.drop(columns=drop)

train_target=data["lifetime"]
train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.33,random_state=0)
	
clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_y)

y_pred=clf.predict(test_X)
print(y_pred)
YTrue=test_y.tolist()
difference=[]
RMSE=0
for i in range(len(y_pred)):
    # print("Prediction Delta",250/y_pred[i]-250/YTrue[i])
    # RMSE+=(250/y_pred[i]-250/YTrue[i])**2
    # plt.scatter(250/y_pred[i],250/YTrue[i])
    print("Prediction Delta",y_pred[i]-YTrue[i])
    RMSE+=(y_pred[i]-YTrue[i])**2
    # plt.scatter(y_pred[i],YTrue[i])

RMSE/=len(y_pred)
RMSE=math.sqrt(RMSE)
print("RMSE",RMSE)