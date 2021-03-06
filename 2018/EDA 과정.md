EDA 과정

1. 데이터 셋 확인
  데이터 셋과 친해지는 관계
  1.1. 변수 확인
- 독립/종속 변수의 정의, 
- 각 변수의 유형(범주형인지 연속형인지)
- 변수의 데이터 타입( date. character, numeric 확인

= 데이터 타입에 따라 모델 fitting 할때 전혀 다른 결과가 나오기 때문에 사전에 변수 타입을 체크하고 잘못 설정되어 있는 경우 이 단계에서 변경해주세요 

1.2. Raw 데이터 확인
(1) 단변수 분석
변수 하나에 대해 기술통계 확인을 하는 단계
HIstogram, boxplot 으로 평균, 최빈값, 중간값 등과 함께각 변수들의 분포를 확인
변수형의 경우 boxplot을 사용해서 빈도수 분포를 체크해주면 된다. 

(2) 이변수 분석
변수 두개 간의 관계를 분석하는 단계
변수의 유형에 따라 적절한 시각화 및 분석 방법을 택하면 됩니다. 

- 연속 연속 ) scatter plot /correlation 분석 
- 범주 범주 ) 누적 막대 그래프 / Chi 제곱 분석
- 번주 연속 ) histogram / 2개(T,Z test) 3개 (ANOVA)  집단별 평균 차가 유의한지 여부 

(3) 셋 이상의 변수  
연속형 변수를 feature engineering 을 통해 범주형 변소루 변환한 후 분석

2. 결측값 처리 (Missing value treatment)
  <결측값 발생 유형>
- 무작위 발생
- 다른 변수와 관계가 있게 발생

<처리 방법 종류>
2.1. 삭제
- 결측값이 발생한 모든 관측지를 삭제하거나(전체삭제)
- 데이터 중 모델에 포함시킬 변수들 중 관측값이 발생한 모든 관측지를 삭제 (부분 삭제)

- 전체삭제 :  간편하지만 관측지가 줄어들어 모델의 유효성이 낮아진다. 
- 부분삭제 : 모델에 따라 변수가 제각각 다르기 때문에 관리 cost가늘어난다.

- 결측값이 무작위로 발생한 경우에 사용해야 한다. 
- 그렇지 않은데 관측치를 삭제한 데이터를 사용하면 왜곡된 모델이 생성될 수 있다. 

2.2 다른 값으로 대체 (평균, 최빈값, 중간값)
- 모든 관측치의 평균 값 등으로 대체하는 일괄 대체 방법
- 범주형 변수를 활용해 유사한 유형의 평균값 등으로 대체ㅔ하는 유사유형대체방법

- 결측 갑의 발생이 다른 변수와 관계까 있는 경우 대체 방법이 유용한 측면은 있지만,
  유사 유형 대체 밥법의 경우 어떤 범주형 변수를 유사한 유형으로 선택할 것인

2.3. 예측값 삽입
- 결측 값이 없는 관측치를 트레이닝 데이터로 사용해서 결측값을 예측하는 모델을 만들고
- 이 모델을 통해 결측값이 있는 관측 데이터의 결측값을 예측하는 방법입니다. 
- Regression / Logistic Regressionn 주로 사용


3. 이상값 처리(Outlier treatment)
- 이상값이란 데이터 / 샘플과 동떨어진관측치로, 모델을 왜곡할 가능성이 있는 관측치를 말합니다. 

이상값 찾아내기
- boxplot histogram , 두 개의 변수 간 이상값 scatter plot
- 두 변수 간 회귀 모형에서 Residual, Studentized residual (Standardized residual, leverage, Cook's D)

이상값 처리하기
3.1. 단순삭제
- 이상값이 사람의 오류에 의해 발생한 경우에는 해당 관측치를 삭제
- 단순 오타, 주관식 설문 등의 비현실 적인 응답
- 데이터 처리과정에서 오류 등에 사용

3.2. 다른 값으로 대체
- 삭제의 방법으로 이상치르 제거하면 관측치의 절대량이 작아지는 문제가 발생
- 다른값으로 대체 / 결측값과 유사한  다른 변수들을 사용해서 예측 모델로 이상값 대체

3.3. 변수화
- 이상값이 자연발생한 경우, 단순삭제나 대체를 통해 수립된 모델은 현상을 잘 설명하지 못할 수 있다.  자연발생적인 이상값의 경우 더 깊은 이해가 필요하다. 
- 새로운 변수를 추가하여 이상값의 설명력을 높일 수 있다. 

3.4. 리샘플링
- 해당 이상값을 분리해서 모델을 만든다. 
- 분석범위를 재설정

3.5. 케이스를 분리하여 분석
- 이상값을 포함한 모델과 제외한 모델을 모두 만들과 각각의 모델에 대한 설명을 단다. 

4. Feature Engineering
  : 기존의 변수를 사용해서 데이터에 정보를 추가하는 일련의 과정
  새로운 관측치나 변수를  추가하지 않고도 기존의 데이터를 보다 유용하게 만든다. 

4.1. Scaling
- 변수의 단위를 변겅하고 싶거나, 
- 변수의 분포가 편향되어 있을 경우
- 변수 간의 관계가 잘 드러나지 않는 경우

- LOG 함수,
- SQUARE ROOT 취하는 방법

4.2. BINNING
- 연속형 변수를 범주형 변수로 만드는 방법
- Grouping , 특별한 원칙 없이 분석가의 이해에 따라 진행

4.2. Transform 
- 기존 존재하는 변수의 성질을 이용해 다른 변수를 만드는 방법

4.3. DUMMY

- 기존 범주형 변수를 one-hot encoding