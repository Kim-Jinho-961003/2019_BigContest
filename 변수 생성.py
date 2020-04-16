#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

import pandas as pd
train = pd.read_csv('train_cloud.csv', encoding = 'cp949')
test = pd.read_csv('test_cloud.csv', encoding = 'cp949')

##### 1. Groupby 변수 생성
### 요일별 지연율
f = (train.query('DLY == "Y"').groupby('SDT_DY')['DLY'].count()/train.groupby('SDT_DY')['DLY'].count()).reset_index()
f = f.rename(columns={"DLY" : "요일별지연율"})

f.sort_values(by ='요일별지연율', ascending = False)

df = pd.merge(df, f, how = 'left', on = 'SDT_DY')

### 공항별 지연율
# 출발공항
f1 = (train.query('DLY == "Y" & AOD == "D"').groupby('ARP')['DLY'].count()/train.query('AOD == "D"').groupby('ARP')['DLY'].count()).reset_index()
f1 = f1.rename(columns={"DLY" : "출발공항별지연율"})
# 도착공항
f2 = (train.query('DLY == "Y" & AOD == "A"').groupby('ODP')['DLY'].count()/train.query('AOD == "A"').groupby('ODP')['DLY'].count()).reset_index()
f2 = f2.rename(columns={"DLY" : "도착공항별지연율"}).fillna(0)

display(f1.sort_values(by ='출발공항별지연율', ascending = False))
display(f2.sort_values(by ='도착공항별지연율', ascending = False))

df = pd.merge(df, f1, how = 'left', on = 'ARP')
df = pd.merge(df, f2, how = 'left', on = 'ODP')

### 항공사별 지연율
f = (train.query('DLY == "Y"').groupby('FLO')['DLY'].count()/train.groupby('FLO')['DLY'].count()).reset_index()
f = f.rename(columns={"DLY" : "항공사별지연율"})

f.sort_values(by ='항공사별지연율', ascending = False)

df = pd.merge(df, f, how = 'left', on = 'FLO')

### 경로별 지연율
train['경로'] = train.ARP
train['경로'] = train['경로'].apply(lambda x: x.replace("ARP",""))+train.ODP
train['경로'] = train['경로'].apply(lambda x: x.replace("ARP","_"))
f = (train.query('DLY == "Y"').groupby('경로')['DLY'].count()/train.groupby('경로')['DLY'].count()).reset_index()
f = f.rename(columns={"DLY" : "경로별지연율"})

f.sort_values(by ='경로별지연율', ascending = False)

df = pd.merge(df, f, how = 'left', on = '경로')

### 편명별 지연율
f = (train.query('DLY == "Y"').groupby('FLT')['DLY'].count()/train.groupby('FLT')['DLY'].count()).reset_index().fillna(0)
f = f.rename(columns={"DLY":"편명별지연율"})

df = pd.merge(df, f, how = 'left', on = 'FLT')

### 공항별 주중/주말 지연율
df['주중/주말']=['주말' if each in ['금','토','일'] else '주중' for each in df['SDT_DY']]

df['공항-주중/주말']=df['ARP']+'-'+df['주중/주말']

f=df.groupby(['공항-주중/주말'])['DLY'].agg([('공항별 주중주말 지연건수','sum')]).reset_index()
f1=df.groupby(['공항-주중/주말'])['DLY'].agg([('공항별 주중주말 운행건수','size')]).reset_index()
f['공항별_주중/주말_지연비율']=(f['공항별 주중주말 지연건수']/f1['공항별 주중주말 운행건수'])

f=f.loc[:,['공항-주중/주말', '공항별_주중/주말_지연비율']]

df = pd.merge(df, f, how = 'left', on = '공항-주중/주말')

### 시간대별 지연건수
f = df.groupby('plan_hour')['DLY'].agg([('시간대별 지연건수','size')]).reset_index().sort_values('시간대별 지연건수',ascending=False)

df = pd.merge(df, f, how = 'left', on = 'hour')

### 출발/도착공항 주 초반/후반 지연율
df['일주초반/후반']=['후반' if each in ['금','토','수','목'] else '초반' for each in df['SDT_DY']]

df['출발공항-초반/후반']=df['ARP']+'-'+df['일주초반/후반']
df['도착공항-초반/후반']=df['ODP']+'-'+df['일주초반/후반']

# 출발공항
f=train.groupby(['ARP','일주초반/후반'])['DLY'].agg([('출발공항 초반후반 지연건수','sum')]).reset_index()
f1=train.groupby(['ARP','일주초반/후반'])['DLY'].agg([('출발공항 초반후반 운행건수','size')]).reset_index()
f['출발공항_초반/후반_지연율']=(f['출발공항 초반후반 지연건수']/f1['출발공항 초반후반 운행건수'])
f['출발공항-초반/후반']=f['ARP']+'-'+f['일주초반/후반']

f=f.iloc[:,['출발공항-초반/후반', '출발공항_초반/후반_지연율']]

df = pd.merge(df, f, how = 'left', on = '출발공항-초반/후반')
df

# 도착공항
f=train.groupby(['ODP','일주초반/후반'])['DLY'].agg([('도착공항 초반후반 지연건수','sum')]).reset_index()
f1=train.groupby(['ODP','일주초반/후반'])['DLY'].agg([('도착공항 초반후반 운행건수','size')]).reset_index()
f['도착공항_초반/후반_지연율']=(f['도착공항 초반후반 지연건수']/f1['도착공항 초반후반 운행건수'])
f['도착공항-초반/후반']=f['ODP']+'-'+f['일주초반/후반']

f=f.iloc[:,['도착공항-초반/후반', '도착공항_초반/후반_지연율']]

df = pd.merge(df, f, how = 'left', on = '도착공항-초반/후반')
df

### 항공사 크기별 지연율
# A: 아시아나항공, # J: 대한항공
df['항공사_크기']=['대형' if each in ['A','J'] else '소형' for each in df['FLO']]

f=train.groupby('항공사_크기')['DLY'].agg([('항공사크기별 지연건수','sum')]).reset_index()
f1=train.groupby('항공사_크기')['DLY'].agg([('항공사크기별 운행건수','size')]).reset_index()
f['항공사크기별_지연율']=(f['항공사크기별 지연건수']/f1['항공사크기별 운행건수'])

f=f.iloc[:,['항공사_크기','항공사크기별_지연율']]

df = pd.merge(df, f, how = 'left', on = '항공사_크기')

df = df.drop(['ARP','ODP','STT','YMD','FLT','date'], axis = 1)
df = df.drop(['SDT_DY','FLO','경로','공항-주중/주말','출발공항-초반/후반','도착공항-초반/후반','항공사_크기'], axis = 1)

##### 2. datetime & str 변수 처리
df['SDT_YY'] = df.SDT_YY.astype('int64')
df['SDT_MM'] = df.SDT_MM.astype('int64')
df['SDT_DD'] = df.SDT_DD.astype('int64')
df['hour'] = df.hour.astype('int64')
df['minute'] = df.minute.astype('int64')
df = pd.get_dummies(df, columns=['AOD','주중/주말','일주초반/후반'])

##### 3. 데이터 저장
train = df.query('SDT_YY < 2019')
test = df.query('SDT_YY == 2019')

train.to_csv('final_train.csv', index = False)
test.to_csv('final_test.csv', index = False)
