#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

import pandas as pd
train = pd.read_csv("AFSNT.csv",encoding="cp949")
test = pd.read_csv("AFSNT_DLY.csv",encoding="cp949")
plan = pd.read_csv("SFSNT.csv",encoding="cp949")

airport = pd.read_csv("공항코드.csv", encoding='cp949',engine='python')
company = pd.read_csv("항공사.csv", encoding='cp949',engine='python')


##### 1. 데이터 통일 
### train & test 변수 통일
train.info()
test.info()

# 변수 통일
train = train.drop(['ATT','REG','IRR', 'DRR','CNR','CNL'], axis = 1)
test = test.drop(['DLY_RATE'], axis = 1)

# 부정기편 제거(test에는 부정기편이 없음)
train = train.query('IRR == "N"')
train = train.drop(['IRR'], axis = 1)

def delay(x):
    if x == "Y":
        return 1
    else:
        return 0
train['DLY2'] = train['DLY'].apply(delay)

### 월별 지연율
a = (train.query('DLY == "Y"').groupby('SDT_MM')['DLY'].count()/train.groupby('SDT_MM')['DLY'].count()).reset_index().rename(columns={"DLY" : "월별지연율"})

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import rc
rc('font', family='malgun gothic')
rc('axes', unicode_minus=False)

plt.bar(a.SDT_MM, a.월별지연율)
plt.xlabel('월')
plt.ylabel('지연율')
plt.title('월별지연율')
plt.show()

### 요인별 지연율
# CO1과 CO2의 경우 A/C 지연으로 사람에 의해 발생하는 지연임으로 제외
a = (train.query('DLY == "Y" & DRR != "C02" & DRR != "C01"').groupby('DRR')['DLY'].count()).reset_index().sort_values(by = 'DLY', ascending = False);a.head(10)

def code(x):
    if x == "C10":
        return "A"
    else:
        return x[0]
a['code'] = a['DRR'].apply(code)

# 날씨에 관한 코드인 A가 가장 많은 것을 확인
a.groupby('code')['DLY'].sum().reset_index().sort_values(by = 'DLY', ascending = False)

train = train.query('SDT_MM == 9 & SDT_DD > 15')

##### 2. 데이터셋 구분(모델링 직전에 구분)
### FLO == M (test 데이터)
a = train.groupby(['FLO'])['DLY2'].mean().reset_index().sort_values(by = 'FLO', ascending = True);a

plt.bar(a.FLO, a.DLY2)
plt.xlabel('항공사')
plt.ylabel('지연율')
plt.title('항공사별 지연율')
plt.show()

display(train.FLO.unique())
display(test.FLO.unique())

test1 = test.query('FLO != "M"')
test2 = test.query('FLO == "M"')

### test에만 존재하는 편명
train.groupby(['FLT'])['DLY2'].mean().reset_index().sort_values(by = 'FLT', ascending = False)

display(train.FLT.nunique())
display(test.FLT.nunique())

a = list(train.FLT.unique())
b = list(test.FLT.unique())

f=[]
for i in b:
    if i in a:
        pass
    else:
        f.append(b)

test3 = test1.query('FLT != @f')

##### 3. 데이터 전처리(3번 과정에서 변수 추가)
### 시간관련 변수
train.SDT_YY = train.SDT_YY.apply(lambda x: str(x))
train.SDT_MM = train.SDT_MM.apply(lambda x: str(x))
train.SDT_DD = train.SDT_DD.apply(lambda x: str(x))
train["YMD"] = train.SDT_YY+"-"+train.SDT_MM+"-"+train.SDT_DD+" "+train.STT
train["YMD"] = pd.to_datetime(train["YMD"])
train['hour'] = train['YMD'].apply(lambda x: str(x.hour))
train['minute'] = train['YMD'].apply(lambda x: str(x.minute))

test.SDT_YY = test.SDT_YY.apply(lambda x: str(x))
test.SDT_MM = test.SDT_MM.apply(lambda x: str(x))
test.SDT_DD = test.SDT_DD.apply(lambda x: str(x))
test["YMD"] = test.SDT_YY+"-"+test.SDT_MM+"-"+test.SDT_DD+" "+test.STT
test["YMD"] = pd.to_datetime(test["YMD"])
test['hour'] = test['YMD'].apply(lambda x: str(x.hour))
test['minute'] = test['YMD'].apply(lambda x: str(x.minute))

train.hour.unique()
test.hour.unique()

train.groupby('hour')['DLY2'].mean().reset_index().sort_values(by='DLY2', ascending = False)

train = train.query('hour != "0" & hour != "23"')

### 경로관련 변수
train['경로'] = train.ARP
train['경로'] = train['경로'].apply(lambda x: x.replace("ARP",""))+train.ODP
train['경로'] = train['경로'].apply(lambda x: x.replace("ARP","_"))

test['경로'] = test.ARP
test['경로'] = test['경로'].apply(lambda x: x.replace("ARP",""))+test.ODP
test['경로'] = test['경로'].apply(lambda x: x.replace("ARP","_"))

display(train.경로.nunique())
display(test.경로.nunique())

a = list(train.경로.unique())
b = list(test.경로.unique())
f=[]
for i in b:
    if i in a:
        pass
    else:
        f.append(b)

a = train.groupby('ARP')['DLY2'].mean().reset_index().sort_values(by='ARP', ascending = True)
b = train.groupby('ODP')['DLY2'].mean().reset_index().sort_values(by='ODP', ascending = True)

a = pd.merge(a, airport, left_on = 'ARP', right_on = '공항코드')
b = pd.merge(b, airport, left_on = 'ODP', right_on = '공항코드')

a.loc[:,['ARP','공항명','DLY2']].sort_values(by='DLY2')

b.loc[:,['ODP','공항명','DLY2']].sort_values(by='DLY2')

train.groupby('경로')['DLY2'].mean().reset_index().sort_values(by='DLY2', ascending = False).head(10)

##### 4. 데이터 저장
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
