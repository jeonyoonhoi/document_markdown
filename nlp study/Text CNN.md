# Text CNN



### 전처리 (NLP Preprocessing)

- 입력값은 형태소 분석기를 통해 전처리된 어휘를 사용한다. 
- 원본 코드는 클린징 처리만 하여 그대로 사용하는데 영어 기준으로 구현되어 있어 한글의 경우 이 코드가 모든 한글을 날려버려서 사용할수 없다. (ㅋㅋㅋㅋ)
- 또한 한글은 조사 등 복잡한 언어 구조상 NLP 전처리 작업이 효율적이기 때문에 미리 처리하도록 한다. 
- 한글로 작성된 원래 논문에도 NLP 전처리를 미리 진행하는 형태로 모식도가 설계되어 있음을 확인할 수 있다. 

![img](https://c1.staticflickr.com/1/330/31620677756_2dc2c91a72_o.png)

2. 

- NLP 전처리를 위해서는 한글 형태소 분석기가 필수적인데 여기서는 카카오 내부에서 사용중인 사내 형태소 분석기인 DHA 를 사용했다. 
- KoNLPy 등 오픈소스 형태소 분석기를 활용하면 유사한 결과를 얻을 수 있을 것이다. 



### 데이터(Data)

- 한국어로 미리 형태소 분석 결과 100 문장을 학습 데이터로 제공했다. 
- 평가데이터는 10개 문장을 제공하여 10% 비율로 구성했다. 
- 당연히 실제로 서비스를 하려면 이보다 훨씬 더 많은 데이터가 필요함
- (서울대논문) 을 보면 대용량 분류를 위해 62만개의 문장을 학습했다고 하며, 결과가 더욱 정확하고 정교해지려면 학습 데이터는 당연히 많을수록 좋다. 
- 또한 원문의 알고리즘은 단어 사전에 없을 경우 동일한 벡터값으로 표현되기 때문에 정확도가 떨어진다. 
- 정확도를 높이기 위해서는 모든 문장의 단어를 커버할 수 있을 정도로 충분히 학습하는게 좋다. 
- 문장 데이터는 어휘로 구분하여 색인하고 0에서 전체 어휘 사이즈 만큼 맵핑한다. 각 문장은 정수 벡터가 된다. 



### 모델(Model)

구현하려는 신경망을 구성하면 아래와 같다

![img](C:\Users\YOONHOI\Documents\document_markdown\nlp study\image\textcnn1)

- 첫 번째 레이어는 단어를 저차원 벡터로 임베드 한다. 
- 그 다음 레이어는 여러 사이즈의 필터를 이용해, 임베드된 단어 벡터에 대해 합성곱 변환을 수행한다. 
- 그 다음 합성곱 레이어의 결과를 긴 피쳐 벡터로 맥스풀링하고
- 드롭아웃 정규화를 추가하고
- 소프트맥스 레이어의 결과로 분류를 수행한다. 

- 김윤 박사 논문 :  w2v사용/ 서울대 논문에 따르면 성능향상에 큰 도움은 X







### 구현(Implementation)

- 원문에서는 텐서플로우를 이용하여 논문의 모델을 실제로 구현하는 방법을 소개하여 매우유용하다. 
- 실제 원문의 내용은 구현을 비롯한 코드 소개에 할애하고 있음



### (1)  Embedding Layer

첫 번째 레이어는 단어 색인을 저차원 벡터 표현으로 매핑하는 임베디드 레이어로, 필수적인 룩업 테이블이다. 앞서 김윤 박사의 논문에는 w2v을 사용했으나 여기서는 단순 룩업테이블을 사용한다. 



### (2) Convolution and Max Pooling Layers

이 부분이 가장 중요한 맥스풀링, 합성곱 레이어를 만드는 부분이다. 

각각 크기가 다른 필터를 사용하여 반복적으로 합성곱 텐서를 생성

이를 하나의 큰피쳐 벡터로 병합한다. 

```
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer 컨볼루션층
        filter_shape = [filter_size, embedding_size, 1, num_filters] 
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        # W : 필터행렬
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # h  : 합성곱 출력에 비선형성(ReLU)를 적용한 결과다. 각 필터는 전체 임베딩을 슬라이드 한다. 
        
        # Max-pooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        # VALID : 엣지 패딩 없이 문장을 슬라이드 하여 
        # [-1, s_l - f_s +1, 1, 1] 크기로 좁은 (narrow) 합성곱을 수행함을 의미한다. 
        pooled_outputs.append(pooled)
        
        # 각 필터사이즈의 맥스풀링출력은
        # [batch_size,1,1,num_filters] 가 되며 이것이 최종 피쳐에 대응하는 마지막 피쳐 벡터다        
        # 모든 풀링 벡터는
        # [batch_size, num_filters_total] 모양을 갖는 하나의 긴 피쳐 벡터로 결합된다. 
        # tf.reshape 에 -1을 사용하여 텐서플로우가 차원을 평평하게 만들도록 한다. 

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
self.h_pool = tf.concat(3, pooled_outputs)
self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
```



### (3) Dropout Layer

- 드롭아웃은 합성곱 신경망의 오버피팅을 방지하는 가장 유명하면서도 흥미로운 방법
- 뉴런의 일부를 확률적으로 비활성화한다. 
- 이는 뉴런의 상호적응을 방지하고 피쳐를 개별적으로 학습하도록 강제한다.
- 사람이 그림을 맞출 때 일부를 손으로 가린 채 특징을 학습하여 맞추도록 하는 방식과 유사하게 동작한다. 여기서는 드롭아웃을 학습 중에는 0.5, 평가중에는 1로 비활성화한다. 

### (4)  Scores and predictions

- 맥스풀링(드롭아웃이 적용된 상태에서)으로 피쳐 벡터를 사용하여 행렬 곱셈을 수행하고 가장 높은 점수로 분류를 선택하는 예측을 수행한다. 
- 원 점수를 정규화 확률로 변환하는 소프트맥스를 적용하지만 최종 예측 결과는 변하지 않는다. 

```
with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
    self.predictions = tf.argmax(self.scores, 1, name="predictions")
```



### (5) Loss and Accuracy

- 점수를 이용해 손실함수를 정의한다.
- Loss : 망에서 발생하는 오류를 나타내는 척도
- 손실을 최소화하는게 목표
- 분류 문제에 대한 표준 손실 함수는 cross-entropy loss 를 사용한다. 

```
# Calculate mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
    self.loss = tf.reduce_mean(losses)
```



### (6) Visualizing the Network

- 딥러닝은 레이어가 매우 복잡해질 수 있기 때문에 시가고하가 중요하다. 텐서플로우에는 텐서보드라는 훌륭한 시각화 도구가 포함되어 있고, 이를 이용해 시각화 할 수 있다. 



### (7) Instantiating the CNN and Minimizing the Loss

TextCNN모델을 인스턴스화한 다음 다음 망의 손실함수를 최적화하는 방법

- 아담 옵티마이저 사용

```
global_step = tf.Variable(0, name = "global_step", trainable = False)
optimizer = tf.train.AdamOptimizer(1e-4)
grads_and_vars = optimizer.compute_gradients(cmm.loss)
train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
```



### (8) Summaries

- 텐서플로우에는 다양한 학습 및 평가 과정을 추적하고 시각화하는 써머리 개념이 있다. 
- 예를 들어 시간 경과에 따른 손실 및 정확도의 변화를 추적하고 싶을 때 레이어 활성화 히스토그램 같은 더 복잡한 부분도 추적할 수있다 .
- 써머리는 시리얼라이즈드 오브젝트이며 써머리라이터를 사용해 디ㅓ스크에 기록한다. 

```
# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs", timestamp))
print("Writing to {}\n".format(out_dir))

# summaries for loss and accuracy
loss_summary = tf.scalar_summary("loss", cnn.loss)
acc_summary = tf.scalar_summary("accuracy",cnn.accuracy)

# Train Summaries
train_summary_op = tf.merge_summary([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir,"summaries","train")
train_summary_wirter = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

# Dev summaries
dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir,"summaries","dev")
dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

```



### (9) Initializing the Variables

- 모델을 학습하기 전에 그래프에서 변수를 초기화 해야 한다. 
- ```initalize_all_variabless``` 함수를 사용했으며 정의한 변수를 모두 초기화하는 편리한 함수다. 
- 현재는 deprecated 된 상태이므로 이에 대한 간단하 패치를 제출해서 머지되었다. 
- 물론 변수의 초기화 프로그램을 수동으로 호출 할 수도 있는데 이는 미리 훈련된 값으로 임베딩을 초기화 하려는 경우에 유용하다



### (10) Defining a single Training Step

- 이제 데이터 배치 모델을 평가하고 모델 파라미터를 업데이트하는 학습단계를 정의한다. 

```
def train_step(x_batch, y_batch):
	# A single training step
	feed_dict = {
        cnn.input_x : x_batch,
        cnn.input_y : y_batch,
        cnn.dropout_keep_prob : FLAGS.dropout_keep_prob
	}
	_,step,summaries,loss,accuracy = sess.run([train_op,global_step,train_summary_op, cnn.loss,cnn.accuracy],feed_dict)
	time_str = datetime.datetime.now().isoformat()
	print("{}: step {}, loss {:g},acc {:g}".format(time_str,step,loss,accuracy))
	train_summary_writer.add_summary(summaries,step)
```



-  평가단계도 학습단계와 유사하게 정의할 수 있다. 
- 학습의 유효성을 검증하기 위해 평가 세트의 손실 및 정확도 평가를 위한 유사한 기능을 작성하며, 차이점은 별도 학습 과정이 필요 없으며 드롭아웃을 비활성화 한다는 점 뿐이다. 



### (11) Training Loop

- 마지막으로 트레이닝 루프를 작성한다.
- 데이터 배치 반복하고
- 주기적으로 모델을 평가하고
- 체크포인팅한다.
- (기본값은 100회당 한번 모델 평가하도록 설정, 조절 가능)



### (12) Visualizing Results in Tensorboard

- 앞서 출력 디렉토리에 써머리를 저장하였는데 텐서보드에 이 위치를 지정하여 그래프와 요약정보를 시각화 할 수 있다. 
- 앞서 여러 차례 언급했든 시가고하는 매우 중요하다. 



학습 결과가 부드럽지 않다 => 더 큰배치를 사용하거나 전체 학습 결과로 평가하면 더부드러우 ㄴ결과 얻는다. 

평가 정확도가 학습 정확도보다 상당히 낮다 ==> 오버피팅 => 더 많은 데이터 다 강한 정규화 더 적은 모델파라미터 필요

드롭아웃으로 인해 학습데이터는 더 낮은 손실, 정확도로 시작도니다ㅏ. 



### (13) Extension and Exercises 

모델 성능 개선 팁

- 사전훈련된 word2vec 벡터를 사용하여 임베딩을 초기화한다. 
- 오버피팅 방지를위해 신경망에 L2정규화를추가하고 드롭아웃 비율을 실험한다. 참고로 콬드에는 이미 있으며 기본적으로 비활성화한 상태..
- 가중치 업데이트 및 레이어 엑션에 대한 히스토그램 써머리르 추가하고 텐서보드에서 시각화한다. 







* 모르는 것 해결
* [1, 2, 2, 1] =ksize (1개,w,h,채널)



### Indexing, Slicing, Joining, Mutating Ops

`torch.``cat`(*tensors*, *dim=0*, *out=None*) → Tensor



```python
'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  ## 주석달기 @jeonyoonhoi
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.FloatTensor

# Text-CNN Parameter
embedding_size = 2 # n-gram
sequence_length = 3
num_classes = 2  # 0 or 1
filter_sizes = [2, 2, 2] # n-gram window
num_filters = 3

# 3 words sentences (=sequence_length is 3)
## 각 sentence에 해당하는 감정 라벨을 지정해준다. 
## 눈치 챘을 수도 있지만 단어는 다 3개의 단어로 지정되어 있다. (단어수가 변경되면 여기서는 에러)
## sequence_length = 3 에 맞춰준 것이다. 어떤 코드들에서는 max length 를 지정해 두고
## 그보다 단어의 수가 작은 문장이 입력되면 빈 칸을 다른 임의의 숫자 (아마 0등 이겠지?) 로 채워주는 전처리를 수행하는 방식을 사용하기도 했다. 이 코드에서는 간단히 구현되어있기 때문에 그냥 나와있음. 텐서플로우나 파이토치에서 api로 그러한 기능을 지닌 함수를 제공하기도 한다. 
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)
'''
이제 토치에서의 이 전처리부분은 익숙해졌다. 
1.2. 문장을 일단 다 붙이고 단어로 자른뒤 중복된 단어를 제거해서 word_list를 만들고
3. 단어 : 인덱스(여기서 인덱스는 단어저장된 순으로 매겨진다. 0~ N-1)
4. vocab_size : 워드딕트 길이만큼~
'''

inputs = []
for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

targets = []
for out in labels:
    targets.append(out) # To using Torch Softmax Loss function
    
    
'''
학습 시킬 데이터 inputㄴ, targets의 형태를 바꿔주는 과정 , 
sentensces, 문장들 리스트에서 sen, 한 문장씩 뽑아내서, 그 문장을 단어로 쪼갠 뒤 , 그 단어를 word_dict[]에 키로 검색하면 리턴되는 인덱스들로 array를 만든다 즉, 단어로 된 문장을 인덱스의 리스트로 만드어준 뒤 inputs에 한 줄씩 append 시켜주는 과정이다. 
targers은 사실 labels를 그대로 복사하는 건데 왜저렇게 하는지는 의문이다.,,
'''

input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))


## 텍스트 cnn 구조를 보잣
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.num_filters_total = num_filters * len(filter_sizes)
        # 총 필터 개수 =  필터 개수 * 필터 사이즈
        # 마지막에 pool하고 난 뒤 flatten 할 때 사용한다. 
        self.W = nn.Parameter(torch.empty(vocab_size, embedding_size).uniform_(-1, 1)).type(dtype)
        # W는 단어 사이즈, 임베딩 사이즈 즉 임베딩된 단어 벡터를 위해 만들어준닷
        self.Weight = nn.Parameter(torch.empty(self.num_filters_total, num_classes).uniform_(-1, 1)).type(dtype)
        # Weight는  필터 개수, class 개수로 만들어준다. 
        
        self.Bias = nn.Parameter(0.1 * torch.ones([num_classes])).type(dtype)
        # Bias는 class 개수로 만들어준다. .

    def forward(self, X):
        embedded_chars = self.W[X] 
        # [batch_size, sequence_length, sequence_length]
        embedded_chars = embedded_chars.unsqueeze(1) 
        # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        '''
        배치크기. 단어개수. 단어개수
        
        채널을 추가해준다. cnn에 맞춰주려고
        '''

        '''
        filter_sizes 여기서는 2,2,2, 였는덴 2,3,4일수도, 
        filter_size를 하나씩 꺼내서 수행한다. 한 번의 반복 = filter한번 거침
        '''
        pooled_outputs = []
        for filter_size in filter_sizes:
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(embedded_chars)
            
            h = F.relu(conv)
            '''
            h = 는 convolution을 수행한 뒤 (필터 사이즈(세로, 임베딩 사이즈(한 단어 임베딩 벡터크기)))
            Relu (activation 함수)를 거쳐 나온 결과이다. 
            '''
            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)
            '''
            스터디 때 이 부분에서 멉춰서 오래 걸렸다. 우리가 이해한 대로 까먹지 말고 적어둬야지 
            일단 nn.MaxPool2d() 라는 것을 mp라는 객체로 불러준다. 
            
            나는 이과정을 그림을 미리 그려준 과정으로, (틀을 만든다. )
            이 다음 mp(h)를 그 만든 틀에 내가 처리하고있는 실제 대상인 h를 흘려보내는 것으로 이해했다. 
            
            MaxPool2d()라는 것 이 사용할때 그냥 객체(인풋) 으로 함수처럼 쓰도록 하는 거였당.
            공식문서에도 그렇게 나와있음 
            
            sequence_length - filter_size +1 은 풀링 된 결과의 칸 수 
            (단여가 10개 필터크기가 2 일때 만약 슬라이딩을 한 칸씩 한다면 )
            
            그리고 pooled에 저장되는것은 mp에 h를 넣어준 뒤
            , permute(인덱스 순서대로 모양 바꿔줌) 함수를 적용한 것이다. 
            
            mp(h)의 결과는 컨볼루셔널 레이어 -> 풀링 -> max 값 뽑은 것들 만든 까지 
            
            permute는  0123 에서 0321로 정렬을 바꿔주는 함수고 
            여기서는 모델 구조에 맞춰주기 위해 차원 순서를 바꿔준 듯. 
            pooled_output은 각 필터에 대해 pooling된 텐서들 리스트이당
            '''
		
        
        
        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) 
        # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total]) 
        # [batch_size(=6), output_height * output_width * (output_channel * 3)]
		'''
		h_pool은 torch.cat으로 pooled_outputs : 텐서들 리스트임, 이걸 붙여줌/묶어줌
		h_pool_flat 은 reshape 해준다. 일렬로 쭉 늘여틀여서 -1 = 한줄짜리로 
		'''
        
        
        model = torch.mm(h_pool_flat, self.Weight) + self.Bias # [batch_size, num_classes]
        return model

model = TextCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 여기는 이제 똑같은거다
# 모델을 생성ㅇ하고 crossentropyloss 함수를 쓰고 adam optimizer를ㄹ 사용한다. 

# Training
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Test
test_text = 'sorry hate you'
tests = [np.asarray([word_dict[n] for n in test_text.split()])]
test_batch = Variable(torch.LongTensor(tests))

# Predict
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")
```





