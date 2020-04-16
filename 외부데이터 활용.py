#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

# 해당 코드는 데이터 셋이 다른 데이터와 형태가 다른 대구 지역의 경우를 대표로 진행합니다.  
# (당시 공모전 진행시에는 변수명을 바꿔가며 총 공항의 개수인 15번 진행하였습니다.)   
# 다른 지역의 경우 일자별 기록이기 때문에 바로 "흐름 살펴보기" 부터 진행하면 됩니다.

import pandas as pd
train = pd.read_csv('train.csv', encoding = 'cp949')
test = pd.read_csv('test.csv', encoding = 'cp949')

### 기상 데이터 로드(2016~19년)
# 공항 데이터
a_16 = pd.read_csv('2016weather_a.csv',encoding = 'cp949')
a_17 = pd.read_csv('2017weather_a.csv',encoding = 'cp949')
a_18 = pd.read_csv('2018weather_a.csv',encoding = 'cp949')
a_19 = pd.read_csv('2019weather_a.csv',encoding = 'cp949')

# 공항 근처 지역 데이터
b_16 = pd.read_csv('2016weather_b.csv',encoding = 'cp949')
b_17 = pd.read_csv('2017weather_b.csv',encoding = 'cp949')
b_18 = pd.read_csv('2018weather_b.csv',encoding = 'cp949')
b_19 = pd.read_csv('2019weather_b.csv',encoding = 'cp949')

# 대구 데이터
d_16 = pd.read_csv('2016_대구.csv',encoding = 'cp949')
d_17 = pd.read_csv('2017_대구.csv',encoding = 'cp949')
d_18 = pd.read_csv('2018_대구.csv',encoding = 'cp949')
d_19 = pd.read_csv('2019_대구.csv',encoding = 'cp949')

a_16.columns
b_16.columns
d_16.columns

### a,b,d 변수 통일화
a = pd.concat([a_16,a_17,a_18,a_19], axis = 0)
a.columns = ['spot', 'date', 'wind', 'cloud', 'temper', 'rain']
a.fillna(0, inplace = True)

b = pd.concat([b_16,b_17,b_18,b_19], axis = 0)
b.columns = ['spot', 'date', 'temper', 'rain', 'wind', 'cloud']
b.fillna(0, inplace = True)

d = pd.concat([d_16,d_17,d_18,d_19], axis =0)
d.columns = ['spot', 'datetime', 'temper', 'rain', 'wind', 'cloud'] 
d.fillna(0,inplace = True); display(d.head(),d.shape)

d_date = d.datetime.str.split(' ',expand = True).rename(columns = {0 : 'date', 1:'time'})
d = pd.concat([d, d_date], axis =1).drop(columns = 'datetime')

df_temper = d.groupby('date')['temper'].agg([('temper_mean','mean')]).reset_index()
df_rain = d.groupby('date')['rain'].agg([('rain_mean','mean')]).reset_index()
df_wind = d.groupby('date')['wind'].agg([('wind_mean','mean')]).reset_index()
df_cloud = d.groupby('date')['cloud'].agg([('cloud_mean','mean')]).reset_index()

temper_df = df_temper.set_index('date')
rain_df = df_rain.set_index('date')
wind_df = df_wind.set_index('date')
cloud_df = df_cloud.set_index('date')

### 흐름 살펴보기
import matplotlib.pyplot as plt 
# Running this code in .py
get_ipython().run_line_magic('matplotlib', 'inline')
# Running this code in .ipynb
%matplotlib inline

plt.plot(temper_df['temper_mean'])
plt.show()

plt.plot(rain_df['rain_mean'])
plt.show()

plt.plot(wind_df['wind_mean'])
plt.show()

plt.plot(cloud_df['cloud_mean'])
plt.show()
# 흐름에 따라 패턴이 보이는 전운량에 대해서만 변수 채택(LSTM 진행)

### Modeling
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping

df = cloud_df
# convert nparray
nparr = df['cloud_mean'].values[::-1].reshape(-1,1)
nparr.astype('float32')

# normalization
scaler = MinMaxScaler(feature_range=(0, 1))
nptf = scaler.fit_transform(nparr)

# split train, test
train_size = int(len(nptf) * 0.9)
test_size = len(nptf) - train_size
train, test = nptf[0:train_size], nptf[train_size:len(nptf)]

# create dataset for learning
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# simple lstm network learning
model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (1, look_back)))
model.add(LSTM(128, activation='relu'))
model.add(RepeatVector(3))
model.add(LSTM(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='mean_squared_error', optimizer= 'nadam')
model.fit(trainX, trainY, epochs=50, batch_size=16, verbose=1,validation_split=0.3)

# make prediction
testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform(testY)
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Train Score: %.2f RMSE' % testScore)

# predict last value (or tomorrow?)
lastX = nptf[-23:]
lastX = np.reshape(lastX, (23, 1, 1))
lastY = model.predict(lastX)
lastY = scaler.inverse_transform(lastY)

# plot
plt.title('%s' % df.columns[0])
plt.plot(testPredict)
plt.plot(testY)
plt.show()

daegu = pd.DataFrame({'date' : test.date.unique(), 'cloud' : pd.Series(list(lastY[8:])).astype('float'),'spot':'대구'})

### train & test 데이터와 병합
airport = pd.read_csv("공항코드.csv", encoding='cp949',engine='python')
air = airport.iloc[:,:2]

# train
a  = pd.merge(a, air, left_on = 'spot', right_on = "공항명").drop(columns = '공항명')
b  = pd.merge(b, air, left_on = 'spot', right_on = "공항명").drop(columns = '공항명')
d  = pd.merge(d, air, left_on = 'spot', right_on = "공항명").drop(columns = '공항명')

a_date = a.datetime.str.split(' ',expand = True).rename(columns = {0:'date',1:'time'})
a = pd.concat([a, a_date],axis = 1).drop(columns = 'datetime')

b_date = b.datetime.str.split(' ',expand = True).rename(columns = {0:'date',1:'time'})
b = pd.concat([b, b_date],axis = 1).drop(columns = 'datetime')

a.date = pd.to_datetime(a.date)
b.date = pd.to_datetime(b.date)
d.date = pd.to_datetime(d.date)

a = a[['cloud','공항코드','date']]
b = b[['cloud','공항코드','date']]
d = d[['cloud','공항코드','date']]

a['year'] = a.date.dt.year
a['month'] = a.date.dt.month
a['day'] = a.date.dt.day

b['year'] = b.date.dt.year
b['month'] = b.date.dt.month
b['day'] = b.date.dt.day

d['year'] = d.date.dt.year
d['month'] = d.date.dt.month
d['day'] = d.date.dt.day

df_cloud = pd.concat([a,b,d], axis = 0)

df_cloud = df_cloud.groupby(['공항코드','date'])['cloud'].agg([('cloud_mean','mean')]).reset_index()

train.SDT_YY = train.SDT_YY.apply(lambda x: int(x))
train.SDT_MM = train.SDT_MM.apply(lambda x: int(x))
train.SDT_DD = train.SDT_DD.apply(lambda x: int(x))

train = pd.merge(train, df_cloud, left_on = ['year','month','day','공항코드'], right_on = ['SDT_YY','SDT_MM','SDT_DD','ARP'])
train = pd.merge(train, df_cloud, left_on = ['year','month','day','공항코드'], right_on = ['SDT_YY','SDT_MM','SDT_DD','ODP'])

train.drop(columns = ['공항코드_x','year_x','month_x','day_x','공항코드_y','year_y','month_y','day_y'], inplace = True)

train = train.rename(columns = {'cloud_mean_x':'a_cloud', 'cloud_mean_y':'d_cloud'})

# test
dae_ = pd.merge(dae, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
won_ = pd.merge(won, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
jin_ = pd.merge(jin, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
po_ = pd.merge(po, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
gun_ = pd.merge(gun, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
gwa_ = pd.merge(gwa, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
chu_ = pd.merge(chu, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
gim_ = pd.merge(gim, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
eus_ = pd.merge(eus, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
mua_ = pd.merge(mua, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
gimp_ = pd.merge(gimp, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
yang_ = pd.merge(yang, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
uls_ = pd.merge(uls, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
jeju_ = pd.merge(jeju, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')
inc_ = pd.merge(inc, air, left_on = 'spot', right_on = '공항명', how = 'left').drop(columns = '공항명')

lst = [dae_,won_,jin_,po_,gun_,gwa_,chu_,gim_,eus_,mua_,gimp_,yang_,uls_,jeju_,inc_]
df_test = pd.concat(lst, axis = 0)

df_test = df_test[['date','cloud','spot','공항코드']]

df_test.date = pd.to_datetime(df_test.date)

test = pd.merge(test, df_test, left_on = ['ARP','date'], right_on = ['공항코드','date']).drop(columns = ['spot','공항코드'])
test = pd.merge(test, df_test, left_on = ['ODP','date'], right_on = ['공항코드','date']).drop(columns = ['spot','공항코드'])
test = test.rename(columns = {'cloud_x' : 'a_cloud', 'cloud_y':'d_cloud'}) ; test.head()

### 데이터 저장
train.to_csv('train_cloud.csv', index = False)
test.to_csv('test_cloud.csv', index = False)
