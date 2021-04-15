import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal
from scipy.optimize import curve_fit 
import math
import pandas as pd 
from sklearn.model_selection import train_test_split
import json 


#Read data from pkl
def GetData():
    batch1 = pickle.load(open(r'.\Data\batch1V.pkl', 'rb'))
    #remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    batch2 = pickle.load(open(r'.\Data\batch2V.pkl','rb'))
    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]
    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]
    del batch2['b2c7']
    del batch2['b2c8']
    del batch2['b2c9']
    del batch2['b2c15']
    del batch2['b2c16']

    batch3 = pickle.load(open(r'.\Data\batch3V.pkl','rb'))
    # remove noisy channels from batch3
    del batch3['b3c37']
    del batch3['b3c2']
    del batch3['b3c23']
    del batch3['b3c32']
    del batch3['b3c38']
    del batch3['b3c39']
    bat_dict = {**batch1, **batch2, **batch3}

    Data=[]
    for i in bat_dict.keys():
        Data.append((bat_dict[i],bat_dict[i]["cycle_life"]))

    Data=sorted(Data,key=lambda x:x[1])

    Processed=[]
    for i,_ in Data:
        Processed.append(i)

    return Processed


def GetSmoothedData(width=51,degree=6):
#Denoising
    Data=GetData()
    for i in Data:
        for j in i['summary']:
            if j != "cycle":
                original=i['summary'][j]
                smoothed=scipy.signal.savgol_filter(original,width,degree)
                i['summary'][j]=smoothed
    return Data

class FeatureExtraction:
    def __init__(self,loadData=True):
        if loadData:
            self.Data=GetSmoothedData()
        else:
            self.Data=[]

        self.I=30
        self.J=[100,150,200,250]
        self.features=[]
        self.lifetime=[]
        self.df=pd.DataFrame()
        self.regDF=pd.DataFrame()

    def pipeline(self):
        self.VoltageRelated()
        self.CapacityRelated()
        self.TemperatureRelated()
        # self.saveData()
        self.TransfertoDF()
        self.df.to_csv('original.csv', encoding='gbk')
        self.RegularDF()
        self.regDF.to_csv('regularized.csv', encoding='gbk')


    #Voltage Related Features
    def VoltageRelated(self): 
        # voltageFeature=[]    
        for batteryidx in range(len(self.Data)):
            #V1: Intensity decrease of IC peak from cycle I to J
            #V2: Voltage Shift of IC peak from cycle I to J
            #V3: Min of QI-J )
            #V4: Max of QI-J 
            #V5: Mean of QI-J
            #V6: Variance of QI-J
            #V7: Skewness of QI-J
            #V8: Kurtosis of QI-J
            if self.Data[batteryidx]["cycle_life"]<250:
                continue 
            OneBattery={}
            OneBattery["V1"]=[]
            OneBattery["V2"]=[]
            OneBattery["V3"]=[]
            OneBattery["V4"]=[]
            OneBattery["V5"]=[]
            OneBattery["V6"]=[]
            OneBattery["V7"]=[]
            OneBattery["V8"]=[]


            Vdlin=self.Data[batteryidx]["Vdlin"]
            maxI,maxI_idx=self.DQDVmax(batteryidx,self.I)
            QI=self.Data[batteryidx]["cycles"][str(self.I)]["Qdlin"]
            for j in self.J:
                maxJ,maxJ_idx=self.DQDVmax(batteryidx,j)
                OneBattery["V1"].append(abs(maxI-maxJ))
                OneBattery["V2"].append(abs(Vdlin[maxI_idx]-Vdlin[maxJ_idx]))
                QJ=self.Data[batteryidx]["cycles"][str(j)]["Qdlin"]
                DeltaQ=QI-QJ

                OneBattery["V3"].append(np.min(DeltaQ))
                OneBattery["V4"].append(np.max(DeltaQ))

                D_mean = np.mean(DeltaQ) #计算均值
                D_var = np.var(DeltaQ)  #计算方差
                D_sc = np.mean((DeltaQ - D_mean) ** 3)  #计算偏斜度
                D_ku = np.mean((DeltaQ - D_mean) ** 4) / pow(D_var, 2) #计算峰度

                OneBattery["V5"].append(D_mean)
                OneBattery["V6"].append(D_var)
                OneBattery["V7"].append(D_sc)
                OneBattery["V8"].append(D_ku)
                
            self.features.append(OneBattery)
            self.lifetime.append(self.Data[batteryidx]["cycle_life"])

            
    def DQDVmax(self,batteryidx,t):
        cycleIData=self.Data[batteryidx]["cycles"][str(t)]
        absDQDV=abs(cycleIData["dQdV"])
        maxI_idx=np.argmax(absDQDV,axis=0)
        maxI=absDQDV[maxI_idx]
        return maxI,maxI_idx

    #Capacity Related Feature
    def CapacityRelated(self):
        outlier=0
        for batteryidx in range(len(self.Data)):
            #p1,p2: Model A: Linear model 
            #p3,p4: Model B: Square root model
            #p5,p6,p7: Model C: coulombic efficiency-based model
            if self.Data[batteryidx]["cycle_life"]<250:
                outlier+=1
                continue 
            OneBattery={}
            OneBattery["p1"]=[]
            OneBattery["p2"]=[]
            OneBattery["p3"]=[]
            OneBattery["p4"]=[]
            OneBattery["p5"]=[]
            OneBattery["p6"]=[]
            OneBattery["p7"]=[]

            QDischarge=self.Data[batteryidx]["summary"]["QD"]
            for j in self.J:
                k=np.arange(self.I,j)
                C=QDischarge[self.I:j]
                #modelA
                popt,_=curve_fit(self.modelA,k,C,maxfev=500000)
                OneBattery["p1"].append(popt[0])
                OneBattery["p2"].append(popt[1])

                #modelB
                popt,_=curve_fit(self.modelB,k,C,maxfev=500000)
                OneBattery["p3"].append(popt[0])
                OneBattery["p4"].append(popt[1])

                #modelC
                popt,_=curve_fit(self.modelC,k,C,maxfev=500000)
                OneBattery["p5"].append(popt[0])
                OneBattery["p6"].append(popt[1])
                OneBattery["p7"].append(popt[2])
            self.features[batteryidx-outlier].update(OneBattery)


    def modelA(self,k,p1,p2):
        return p1*k+p2
    def modelB(self,k,p3,p4):
        return p3*np.sqrt(k)+p4
    def modelC(self,k,p5,p6,p7):
        return p5*(p6**k)+p7
    
    #Temperature Related Feature
    def TemperatureRelated(self):
        outlier=0
        for batteryidx in range(len(self.Data)):
            #T1 Avg of TI-J
            #T2 Max of TI-J
            #T3 Min of TI-J
            if self.Data[batteryidx]["cycle_life"]<250:
                outlier+=1
                continue 
            OneBattery={}
            OneBattery["T1"]=[]
            OneBattery["T2"]=[]
            OneBattery["T3"]=[]

            TI=self.Data[batteryidx]["cycles"][str(self.I)]["Tdlin"]
            for j in self.J:
                TJ=self.Data[batteryidx]["cycles"][str(j)]["Tdlin"]
                DeltaT=TI-TJ
                OneBattery["T1"].append(np.mean(DeltaT))
                OneBattery["T2"].append(np.min(DeltaT))
                OneBattery["T3"].append(np.max(DeltaT))
            self.features[batteryidx-outlier].update(OneBattery)

    # #Save as Json File
    # def saveData(self,filename=("Features.json","Lifetime.json")):
    #     with open(filename[0],'w') as file_obj:
    #         json.dump(self.features,file_obj)
    #     with open(filename[1],'w') as file_obj:
    #         json.dump(self.lifetime,file_obj)
    
    # #Load Json File to features and lifetime
    # def loadData(self,filename=("Features.json","Lifetime.json")):
    #     with open(filename[0]) as file_obj:
    #         self.features=json.load(file_obj)
    #     with open(filename[1]) as file_obj:
    #         self.lifetime=json.load(file_obj)
    
    #Convert the Dictionary to DataFrame, the lifePercent is the percent of 250 cycles in the whole life
    def TransfertoDF(self):
        for battery,lifetime in zip(self.features,self.lifetime):
            f={}
            for i in battery:
                for j in range(len(battery[i])):
                    k=i+"-"+str(j)
                    f[k]=battery[i][j].squeeze()
                    # else:
                        # f[k].append(battery[i][j].squeeze())
            f["lifePercent"]=250/lifetime.squeeze()
            f["lifetime"]=lifetime.squeeze()

            self.df=self.df.append(f,ignore_index=True)
        # print(df)

    #Use Min-Max scaler to regular the data excluding the lifePercent/lifetime, which will served as label
    def RegularDF(self):
        newDataFrame = pd.DataFrame(index=self.df.index)
        columns = self.df.columns.tolist()
        for c in columns:
            d = self.df[c]
            MAX = d.max()
            MIN = d.min()
            d=d.tolist()
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).squeeze()

        newDataFrame["lifePercent"]=self.df["lifePercent"]
        newDataFrame["lifetime"]=self.df["lifetime"]

        self.regDF=newDataFrame

if __name__=="__main__":
    Test = FeatureExtraction()
    Test.pipeline()
