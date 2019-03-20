## A Convolutional Neural Netwrok for Modelling Sentences



#### Convolutional Neural Networks with Dynamic kk-Max Pooling

- wide-convolution을 k-max pooling을 사용하는 dynamic pooling layer로 대체한다. 
- 이 모델의 input이 가변길이 sentence를 받기 때문에 중rks layer 에서의 feature map 의 넓이는 각각다르다
- 하지만 CNN의 경우 고정 크기의 map에 대해서 연산할 수 있다. 
- 이러한 문제를 해결하기 위한 것이 Dynamic Convolutional Neural Network 다. 

![dcnn](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\dcnn1.jpg)



### Wide Convolution

- 주어진 input 문장에 대해서 각 단어의 임베딩값을 구한다.
- 만들어진 임베딩 벡터를 concatenate 해서 문장 matrix를 만든다. 
- 여기서 각 단어 벡터 wi는 학습 과정에서 최적화 될 것이다.
- convolutional layer는 weight matrix 인 m∈R^(d×m) 를 convolution한다.
- 여기서 dimension d와 필터의 넓이인 m 은 hyper parameter 이다. 
- 이 과정을 wide one-dimensional convolution이라 한다. 
- 그리고 이 결과 나오는 matrix를 c 라고 하고
- dimension 은 d X ( s + m - 1)이 된다. 



### k-Max Pooling

- 여기서 사용할 Pooling연산은 Max-TDNN 문장 모델에서 사용된 max pooling over the time dimension을 사용한다. 
- 이 연산은 일반적인 일부 지역을 모아 pooling 하는 것과는 다른 연산이다.
- 과정에 대해 설명하면 
- 우선 어떤 k값이 주어진다. 
- p>=k인 dimension을 가지는 sequence(vector) P에 대해서 k-max pooling은 sequence P 에서 k개의 최대값을 선택해 subsequence P k_max를 만든다
- k-max pooling은 위치가 가까운 것에 신경쓰지 않고 P에서 k 개의 active한 값을 뽑아낸다.
- 연산 뒤에는 각 feature의정확한 위치정보는 소실되지만 각 feature들간의 순서정보는 보존된다. 
- 이 k-max pooling연산은 해당 네트워크에서 가장 위에 있는 convolutional layer이후에 적용된다 
- 이 연산을 통해 jully connected layer 로 가기 위한 input값의 크기 제한이 없어진다. 
- 다음으로 우리가 알아볼 pooling은 convolutional layer중간에서 적용되며 k값이 고정되어 있지 않고 더 넓은 범위의 순서를 보존하며 뽑기 위해 동적으로 선택된다. 



### Dynamic k-Max Pooling

- dynamic k-max pooling연산은 k-max pooling과 같은 과정으로 수행되지만 k 값이 정해지는 함수가 존재한다. 
- 이 함수는 문장의 길이와 네트워크의 총 깊이(전체 레이어의 수)에 의해 결정된다. 
- 문장 길이를 s라 하고 네트워크의 깊이는 L이라 한다. 
- 여기서 l은 현재 convolution layer의 층
- k_top 은 가장 높은 층의 convolutional layer를 위한 고정된 parameter 이다.
- 전체 3개의 convolutional layer가 있고 
- k_top = 3이고 input 문장의 길이는 s = 18 이라 하자
- 이 pooling에 대해서 조금 직관적으로 이해를 해보자.
- k값은 층이 올라갈 수록 수가 적어지는 구조이다. 
- 이 구조가 의미하는 것은 층이 낮은 곳에서는 더 많은 중요한 feature를 뽑고 층이 올라갈수록 더욱 중요한 feature를 고르는 과정으로 이해할 수 있다. 



### Non-linear Feature Function

- 위의 k-max pooling연산은 convolution 연산의 결과에 적용된다는 것을 알 수 있다. 
- 이렇게 pooling을 한 이후에 바로 값을 다음으로 보내는 것이 아니라 bias를 더하고 non-linear 함수인 g를 요소별로 적용 시킨 후 적용시킨 후 다음 층으로 전달된다. 
- 각 pooling matrix에서는 각각의 단일 값의 bias가 존재한다. 



### Multiple Feature Maps

- 이때까지 우리는 각 layer에서 convolution과 pooling이 하나씩만 적용되는 경우로 생각했다. 하지만 처음에 봤던 전체 network에 대한 그림을 보면 알 수 있듯이 각 layer에서 2개씩 적용된다. 
- 즉 여러개의 feature map을 만든 뒤 feature map을 새로운 parameter를 곱한 뒤 합쳐주는 형태이다. 
- *이 의미하는 것은 wide convolution이다. 실제 network에서 병렬로 진행하는 모델의 수는 사용자가 정의한다. (2개가 아니어도 됨)



### Folding

- 이때까지의 모델을 살펴보면, feature map을 뽑을 때 convolution 과 pooling이 모두 각 row에 대해서 연산되는 것을 볼 수 있다. 
- 이렇게 진행을 한다면 각 row는 독립적으로 진행되어서 서로 연관관계에 대한 정보는 전혀 포함 할 수 없다. 
- 따라서 이러한 row-independent 한 문제를 해결하기 위해 연산을 마친 각 row들을 합치는 folding과정을 마지막 pooling전에 수행한다. 
- 2개의 row를 각각 합하는 형태로 진행되기 때문에 새로운 parameter도 필요하지 않다. 
- 이 과정을 수행하면 matrix의 높이가 절반으로 줄어들게 된다. 



## Properties of sentence Model

해당 네트워크의 sentence modeling의 몇 가지 특성에 대해서 소개한다. 

### Word and n-Gram Order

- sentence modeling에서 중요한 것은 특정 n-gram 이 input에서 발생하는 것을 구별하는 것과 구별한 n-gram 에 대해서 상대적인 순서 정보 또한 가지고 있는 것이 중요하다. 
- 해당 network는 두 중요점을 위해 설계했다.
- m필터로 wide convolution을 하면 특정n-gram에 대하 ㄴ정보를 뽑고
- pooling과정에서 각 순서정보를 포함하게 된다. 



### Induced Feature Graph

- 해당 모델을 좀 더 큰 그림으로 보자
- 우선 각 matrix의 하나의 row의 관점에서만 보게 된다면 각 값들 중 몇 가지만 선택되어서 다음 layer로 전달된다. 
- 나머지 값들은 탈락된 것이며 선태고디 ㄴ값들 중 그 다음으로 또 전달되는 것은 그 일부이다. 
- 이러한 내용을 전체적으로 보면 각 값(feature)에 대한 graph 구조로 생각할 수 있다. 
- 이 글의 시작에서 소개한 그림의 형태로 진행되고 있는 것이다. 

![klo](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\9T5tSru.jpg)

## Train

학습에 사용된 방법들에 대해 소개하면 다음과 같다. 

- top-layer는 softmax를 통해 예측한 class에 대한 확률분포를 가진다. 
- 학습과정에서 이 분포에 대해 실제 분포와 비교해서 cross-entropy를 최소화하는 방향으로 학습한다. 
- 학습시 L2정규화를 사용했다. 
- mini-batch 방식으로 학습시켰다. 
- update policy는 Adagrad를 통해  update했다. ㄷ



## Experiments

실험은 다음의 4가지 문제에 대해서 진행했다. 

- Sentiment Prediction in Movie Review
- Question Type Classification
- Twitter Sentiment Prediction with Distance Supervision
- Visualising Feature Detectors

각 task들에 대한 결과는 다음과 같다. 



