# Word2Vec

## Ref 

1. [ **딥 러닝을 이용한 자연어 처리 입문**](https://wikidocs.net/book/2155)
2. [09. 예측 기반의 단어 표현(Pred ...](https://wikidocs.net/22644)
3. [1) 워드투벡터(Word2Vec)](https://wikidocs.net/22660)



## 1) 워드투 벡터(Word2Vec)

- 원-핫 인코딩은 단어 간 유사성을 계산할 수 없다는 단점을 가짐

  : 벡터에 단어의 의미가 들어가 있지 않기 떄문이다.

  ==> 단어간 유사성을 고려하기 위해 :  단어의 의미를 벡터화 하자!

- [사이트 주소](http://w.elnn.kr/search/)



### 1. 희소 표현(Sparse Representation)

- 원-핫 인코딩을 통해서 나온 원-핫 벡터들은 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현되는 벡터 표현 방법이었음. 

- 희소표현이란 : 벡터, 행렬 의 값의 대부분이 0으로 표현되는 방법 

- 따라서 원-핫벡터 = 희소백터(Sparse Vector)

- 이러한 표현 방법은 단어간 유사성을 표현할 수 없다는 단점을 지님

- 그에 대한 대안으로 단어의 '의미'를 다차원 공간에 벡터화 하는 방법을 찾게되었음 ==> 분산표현

- 임베딩 작업 : 분산표현을 이용하여 단어의 의미를벡터화 하는 작업

- 임베딩 벡터 : 이렇게 표현된 벡터 

  

### 2. 분산표현(Distributed Representation)

- 분포 가설(Distributional Hypothesis) : 비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다. 

- 분산표현은 분포가설가정을 이용하여 코퍼스로부터 단어들의 데이터셋을 학습하고, 벡터에 단어의 의미를 여러 차원에 분산하여 벡터로 표현합니다.

- 이렇게 표현된 벡터들은 ㄱㄷ이 벡터의 크기가 단어집합의 크기일 필요없음. 

- 원-핫 벡터로 표현할 때에는 갖고 있는 코퍼스에 단어가 10000개엿다면 벡터의 차원은 10000이어야만 했다. (심지어는 인덱스에해당하는 부분만 1, 나머지는 0) 

- 단어 집합이 클수록 그만큼 크기의 고차원의 벡터가 되는것이다. 

- 워드투벡터는 차원수가 이렇게 필요 없고 사용자가 설정한 차원의 크기만 갖게 된다. 더이상 1과 0으로 이루어진 벡터도 아니다. 

  Ex) 강아지 = [0.2 0.3 0.5 0.7 0.2 ... 중략 ... 0.2]

##### 요약

희소표현 : 고차원에 각 차원이 분리된 표현 방법

분산표현 : 저차원에 단어의 의미를 표현하되, 여러 차원에다 분산하여 표현한다는 것 / 단어간 유사도를 계산할 수 잇다. 

이를 위한 학습 방법에는 과거에는NNLM, RNNLM이 있었으나 현재에 와서는 해당 방법들의 속도를 대폭 개선시킨 워드투벡터(Word2Vec)가 쓰이고 있다. 

워드 투 벡터 방식에는 CBOW와 Skip-Gram 두 가지의 방식이 존재한다. 

cbow는 주변에 있는 단얻르을 가지고 중간에 있는단어를 예측하는 방법

skip-gram은 중간에 있는 단어를 갖고 주변 단어들을 예측하는 방법



### 3.  (방식 1) CBOW(Continuous Bag of Words)

- 윈도우크기 :  중심단어를 예측하기 위해 앞, 뒤로 몇 개의 단어를 볼지 결정했다면 이 범위를 window 라고 한다. 
- 중심단어(center word), 주변단어(context word)

- 슬라이딩 윈도우(Sliding window) : 윈도우 크기르 ㄹ정했다면 윈도우를 계속 움직여서 주변 단어와 중심단어 선택을 바꿔가며 학습을 위한 데이터 셋을 만들 수 있는데 이것을 슬라이딩 윈도우 라고 한다. 



![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\nn.PNG)

input layer 의 입력으로서 앞, 뒤로 사용자가 정한 윈도우 크기 범위 안에 있는 주변 단어들의 원 핫 벡터가 들어가게 되고,  output layer에서 예측하고자 하는 중간단어의 원 핫 벡터가 필요합니다. 

Word2Vec의 학습을 위해서 중간 단어의 원 핫 벡터가 필요합니다. 

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\cbow_nn)

두 가지에 주목하자

1. hidden layer의 크기가 N : (임베딩하고 난 벡터의 크기)

2. 입력층과 은닉층 사이의 가중치 W는 V x N행렬, W'는 N x V 행렬

   여기서 V는 단어 집합의 크기를 의미한다.

   이 두 행렬은 동일한 행렬을 Transpose한 것이 아니라 서로 다름

   뉴럴 넷 훈련 전에 이 가중치행렬 W 와 W' 는 대게 굉장히 작은 랜덤 값을 가지게 됩니다. CBOW 는 주변 단어로 중심 단어를 더 정확히 맞추기 위해 계속해서 이 W와 W'를 학습해가는 구조입니다. 

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\img3)

- 입력벡터는 원핫벡터이기 떄문에 i번째 인덱스어 1이라는 값을 가지고 나머지는 0을 가지는 입력벡터와 가중치 W행렬의 곱은, 사실 W행렬의 i번째 행을 그대로 읽어오는 것과 (lookup) 동일합니다. 
- 이 작업을 Table lookup이라고도 부른다.
- CBOW의 목적은 W와 W'을 잘 훈련시키는 것이라고 언급한 적이 있는데 사실 그 이유가 여기서 lookup해온 W의 각 행벡터가 사실 W2V을 수행한 후 각 단어의 N차원의 크기를 갖는 임베딩 벡터이기 때문



![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\img4)

- 이렇게 각 주변 단어의 원 핫 벡터에 대해서 가중치 W가 곱해서 생겨진 결과 벡터들은 은닉층에서 만나 이 벡터의 평균인 벡터를 구하게 됩니다. 
- 만약 윈도우 크기가 2라면, 입력 벡터의 총 갯수는 2m 이기 때문에 중간 단어를 예측하기 위해선 총 4개가 입력 벡터로 들어간다
- 이 부분이 skip gram 과의 차이점이 된다. (skip gram은 입력이 중심 단어 하나이기 때문에 은닉층에서 벡터의 평균을 구하지 않는다.)

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\img5)

- 이렇게 구해진 평균 벡터는 두 번째 가중치 행렬인 W'와 곱해집니다. 이렇게 되면 크기가 V와 동일한 벡터 (인풋이었던 원핫벡터들과 차원 동일)나오게 됩니다.
- 이 벡터에 CBOW는 ㄴsoftmax를 취한다. ==> 총 원소의 합은 1, 출력값은 0~1사이 ==> 이걸 스코어 벡터(score vector)라고 한다. 
- 스코어 벡터의 j번쨰 인덱스가 가진 0과 1사이의 값은 j번쨰 단어가 중심 단어일 확률을 나타냅니다. 
- 이 스코어 벡터는 우리가 실제로 값을 알고있는 벡터인 중심 단어 원 핫 벡터의 값에 가까워져야 합니다. 스코어 벡터를 y^ 라고 하겠습니다. 중심단어를 y라고 했을 때 이 두 벡터가 가까워지게 하기 위해서 cbow는 corss entropy 함수를 사용합니다. 즉, 다른말로 loss function으로 cross-entropy 함수를 사용합니다. 

- 식으로 쓰면 다음과 같은데, 원 핫 벡터라고 하면 아래 식으로 간소화 할 수 있다. 

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\img)

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\crossentrophy2.PNG)

- 이 식이 왜 loss function으로 적합할까? c를 중심단어에서 1을 가진 차원의 값의 인덱스라고 한다면 y^c =1은 y^가 y를 정확하게 예측한 경우가 됩니다. 

- 이를 식에 대입하면 -1log(1)=0이 되기 때문에 이떄의 loss = 0 이 됩니당

- 즉 우리는 

- ![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\crossentrophy.PNG)

  ![img](https://wikidocs.net/images/page/22660/crossentrophy.PNG)

  이 값을 최소화 하는 방향으로 학습해야 합니다.

- 이제 Back Propagation을 수행하면 W와 W' 가 학습이 되는데 학습이 다 되었다면 N차원의 크기를 갖는 W행이나 W'의 열로부터 어떤 것을 임베딩 벡터로 사용할지 결정하면 됩니다. 떄로는 W와 W' 의 평균치를 가지고 임베딩 벡터를 선택하기도 합니다.



### 4. (방식2) Skip-gram

- Skip gram 은 중심단어에서 주변 단어를 예측하려고 합니다. 

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\skipgram.png)

- 중심 단어에서 주변 단어를 예측하기 때문에 hidden layer에서 벡터들의 평균을 구하는 과정은 없다. 
- 여러 논문에서 성능 비교를 했을 때 전반적으로 Skipgram이 cbow보다 성능이 좋다고 알려져 있음



### 5. 네거티브 샘플링(Negative Sampling)

- 요즘 워드투벡터를 사용한다고하면 SGNS (Skip Gram with Negative sampling)을 사용하는 것이 보통입니다. 
- 근본적 이유는 계산량을 줄이기 위함이다. 
- 모든단어에 대해 softmax를 계산하지 않고 일부에 대해서만 softmax계산을 할 수 있는 방법이 있다면 계산량이 줄어들지 않을까.



### 6. 영어 Word2vec 만들기



