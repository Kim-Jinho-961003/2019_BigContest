### Competition Topic
https://bigcontest.or.kr/points/content.php

항공 운항 데이터를 활용한 "항공 지연 예측" futures league  
2016~2018년 간의 데이터를 통해 2019년 9월 16일 ~ 30일 간의 항공편별 지연 여부를 예측하는 항공지연 예측 모형 개발    
평가데이터 기반의 항공편별 지연여부 예측 정확도  
평가지표 : roc_auc 

### Team Members
Kookmin Univ 
Bigdata Business Stastics Major
1. Kim. Jinho 
2. Kim. Taeuk
3. Bae. yuna
4. Jeong. jaeyeob

Current process (full)
----------------------
1. EDA:   
train 데이터와 test 데이터의 형태가 다르기 때문에 train에서만 가지고 있는 변수, 시간대 등의 부분을 제거하고 test에서만 가지고 있는 value를 파악해 각각 다른 데이터 set로 구분하였습니다. 그리고 통일된 데이터를 통해 전처리 및 EDA를 진행하여 지연에 유의한 변수들을 확인합니다.
2. 외부데이터 활용:   
지연에 가장 영향력 있는 날씨를 반영하기 위해 2016~2019년 기상 정보를 가져왔으며, 미래 예측에 해당되는 test 데이터 기간에는 LSTM을 통해 날씨를 예측하여 데이터에 포함
3. Feature Engineering:   
범주형 변수가 많아 ~의 지연율이라는 수치적 대응을 통해 정보 손실을 줄임 (Groupby 변수 생성)
4. Modeling :  
지연의 경우가 현저히 적기 때문에 imbalance learning을 진행하고 다양한 모델에 대한 테스트 및 선택된 모델에 대한 파라미터 튜닝 진행. 
5. Ensemble :   
soft_voting과 gmean을 통해 앙상블 진행
