import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import  OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt 
import math 
data=pd.read_csv(r"regularized.csv",low_memory=False)

drop=["Unnamed: 0","lifetime","lifePercent"]
TemperatureRelated=["T1-0","T1-1","T1-2","T1-3","T2-0","T2-1","T2-2","T2-3","T3-0","T3-1","T3-2","T3-3"]
train_data=data.drop(columns=drop)
# train_data=train_data.drop(columns=TemperatureRelated)

train_target=data["lifePercent"]
train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.2,random_state=0)
data_test = xgb.DMatrix(test_X, label=test_y)
data_train = xgb.DMatrix(train_X, label=train_y)

watch_list = [(data_test, 'eval'), (data_train, 'train')]

param = {'eta': 0.02, 'objective': 'reg:squarederror','num_boost_round':1100}
bst = xgb.train(param, data_train, num_boost_round=1000, evals=watch_list)
y_pred = bst.predict(data_test)
pickle.dump(bst, open("pima.pickle.dat", "wb"))

YTrue=test_y.tolist()
difference=[]
RMSE=0
for i in range(len(y_pred)):
    print("Prediction Delta",250/y_pred[i]-250/YTrue[i])
    RMSE+=(250/y_pred[i]-250/YTrue[i])**2
    # plt.scatter(250/y_pred[i],250/YTrue[i])
    # print("Prediction Delta",y_pred[i]-YTrue[i])
    # RMSE+=(y_pred[i]-YTrue[i])**2
    # plt.scatter(y_pred[i],YTrue[i])

RMSE/=len(y_pred)
RMSE=math.sqrt(RMSE)
print("RMSE",RMSE)
# plt.xlabel("Predicted Life Cycle")
# plt.ylabel("True Life Cycle")
# plt.show()
xgb.plot_importance(bst)
plt.show()