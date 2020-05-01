import pandas as pd
import numpy as np
import math
import os

file_name=''
file_size=0
X=[]
Y=[]
TS=[]
BS=[]
AZ=[]
AL=[]
P=[]
V=[]
SDX=[]
SDY=[]
A=[]
SDV=[]
SDA=[]
aX=0
aY=0
aV=0
aA=0
        
file_name='U1S1.txt'
file=pd.read_csv(file_name,delimiter=' ',names=['X','Y','TS','BS','AZ','AL','P'],header=None,skiprows=1)
file_size=len(file)
X=file['X']
Y=file['Y']
TS=file['TS']
BS=file['BS']
AZ=file['AZ']
AL=file['AL']
P=file['P']
aX=sum(X)/file_size
aY=sum(Y)/file_size
for k in range(0,file_size-1):
    if TS[k]==TS[k+1]:
        X[k+1]=(X[k]+X[k+1])/2
        Y[k+1]=(Y[k]+Y[k+1])/2
        TS[k+1]=(TS[k]+1)
        BS[k+1]=(BS[k]+BS[k+1])/2
        AZ[k+1]=(AZ[k]+AZ[k+1])/2
        AL[k+1]=(AL[k]+AL[k+1])/2
        P[k+1]=(P[k]+P[k+1])/2
    if k<file_size:
        V.append(((math.sqrt((X[k+1]-X[k])**2+(Y[k+1]-Y[k])**2))*(TS[file_size-1]-TS[0]))/(TS[k+1]-TS[k]))
    SDX.append((X[k]-aX)**2)
    SDY.append((Y[k]-aY)**2)
SDX.append((X[file_size-1]-aX)**2)
SDY.append((Y[file_size-1]-aY)**2)
V.append(0)
data={'X':X,'Y':Y,'TS':TS,'BS':BS,'AZ':AZ,'AL':AL,'P':P,'V':V,'SDX':SDX,'SDY':SDY}
df=pd.DataFrame(data)

    
file=df
file_size=len(file)
X=file['X']
Y=file['Y']
TS=file['TS']
BS=file['BS']
AZ=file['AZ']
AL=file['AL']
P=file['P']
V=file['V']
SDX=file['SDX']
SDY=file['SDY']
for k in range(0,file_size):
    if k<file_size-1:
        A.append(((abs(V[k+1]-V[k]))*(TS[file_size-1]-TS[0]))/(TS[k+1]-TS[k]))
A.append(0)
data={'X':X,'Y':Y,'TS':TS,'BS':BS,'AZ':AZ,'AL':AL,'P':P,'V':V,'SDX':SDX,'SDY':SDY,'A':A}
df=pd.DataFrame(data)


file=df
file_size=len(file)
X=file['X']
Y=file['Y']
TS=file['TS']
BS=file['BS']
AZ=file['AZ']
AL=file['AL']
P=file['P']
V=file['V']
SDX=file['SDX']
SDY=file['SDY']
A=file['A']
aV=sum(V)/file_size
aA=sum(A)/file_size
for k in range(0,file_size):
    SDV.append((V[k]-aV)**2)
    SDA.append((A[k]-aA)**2)
data={'X':X,'Y':Y,'TS':TS,'BS':BS,'AZ':AZ,'AL':AL,'P':P,'V':V,'SDX':SDX,'SDY':SDY,'A':A,'SDV':SDV,'SDA':SDA}
df=pd.DataFrame(data)




avgX=[]
avgY=[]
avgSDX=[]
avgSDY=[]
avgV=[]
avgA=[]
avgSDV=[]
avgSDA=[]
avgP=[]
maxV=[]
maxA=[]
maxP=[]
pen_down=[]
pen_up=[]
pen_ratio=[]
sign_width=[]
sign_height=[]
width_height_ratio=[]
total_sign_duration=[]
range_pressure=[]
range_velocity=[]
range_acceleration=[]




file=df
file_size=len(file)
X=file['X']
Y=file['Y']
TS=file['TS']
BS=file['BS']
AZ=file['AZ']
AL=file['AL']
P=file['P']
V=file['V']
SDX=file['SDX']
SDY=file['SDY']
A=file['A']
SDV=file['SDV']
SDA=file['SDA']
avgX.append(sum(X)/file_size)
avgY.append(sum(Y)/file_size)
avgSDX.append(sum(SDX)/file_size)
avgSDY.append(sum(SDY)/file_size)
avgV.append(sum(V)/file_size)
avgA.append(sum(A)/file_size)
avgSDV.append(sum(SDV)/file_size)
avgSDA.append(sum(SDA)/file_size)
avgP.append(sum(P)/file_size)
maxV.append(max(V))
maxA.append(max(A))
maxP.append(max(P))
pen_down.append(sum(BS))
pen_up.append(file_size-sum(BS))
pen_ratio.append((sum(BS))/(file_size-sum(BS)))
sign_width.append(max(X)-min(X))
sign_height.append(max(Y)-min(Y))
width_height_ratio.append((max(X)-min(X))/(max(Y)-min(Y)))
total_sign_duration.append(TS[file_size-1]-TS[0])
range_pressure.append(max(P)-min(P))
range_velocity.append(max(V)-min(V))
range_acceleration.append(max(A)-min(A))
print(avgX[:2],len(avgX))
print(avgY[:2],len(avgY))
print(avgSDX[:2],len(avgSDX))
print(avgSDY[:2],len(avgSDY))
print(avgV[:2],len(avgV))
print(avgA[:2],len(avgA))
print(avgSDV[:2],len(avgSDV))
print(avgSDA[:2],len(avgSDA))
print(avgP[:2],len(avgP))
print(maxV[:2],len(maxV))
print(maxA[:2],len(maxA))
print(maxP[:2],len(maxP))
print(pen_down[:2],len(pen_down))
print(pen_up[:2],len(pen_up))
print(pen_ratio[:2],len(pen_ratio))
print(sign_width[:2],len(sign_width))
print(sign_height[:2],len(sign_height))
print(width_height_ratio[:2],len(width_height_ratio))
print(total_sign_duration[:2],len(total_sign_duration))
print(range_pressure[:2],len(range_pressure))
print(range_velocity[:2],len(range_velocity))
print(range_acceleration[:2],len(range_acceleration))



data={'avgX':avgX,'avgY':avgY,'avgSDX':avgSDX,'avgSDY':avgSDY,'avgV':avgV,'avgA':avgA,'avgSDV':avgSDV,'avgSDA':avgSDA,
      'avgP':avgP,'maxV':maxV,'maxA':maxA,'maxP':maxP,'pen_down':pen_down,'pen_up':pen_up,'pen_ratio':pen_ratio,
      'sign_width':sign_width,'sign_height':sign_height,'width_height_ratio':width_height_ratio,
      'total_sign_duration':total_sign_duration,'range_pressure':range_pressure,'range_velocity':range_velocity,
      'range_acceleration':range_acceleration}

df=pd.DataFrame(data)

dataset=pd.read_csv('Features.csv')
dataset=dataset.drop({'F','ID'},axis=1)
df=(df-dataset.min())/(dataset.max()-dataset.min())
print(df)