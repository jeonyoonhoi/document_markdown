



# BOAZ CS스터디 계획서

## REF

[CS231n](http://cs231n.stanford.edu/2017/syllabus)

[모두의 딥러닝](https://hunkim.github.io/ml/)



## 시간 및 장소

* 2018 여름방학 ~ 9월 말
* 매주 일요일 2:00~
* 종각/ 안암 / 혜화 



## 1. Stanford CS231n

#### 계획

* 매주 강의 2강씩 듣기 ==> 강의 정리한 뒤 공유 
* 그에 맞는 과제 해온 뒤 공유 



#### Schedule and Syllabus

| Event Type     |   마감   | Description                                                  | Course Materials                                             |
| -------------- | :------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Lecture 1      | 18-08-05 | **Course Introduction**  Computer vision overview  Historical context  Course logistics | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture1.pdf) [[video\]](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 2      | 18-08-05 | **Image Classification**  The data-driven approach  K-nearest neighbor  Linear classification I | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf) [[video\]](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [[python/numpy tutorial\]](http://cs231n.github.io/python-numpy-tutorial) [[image classification notes\]](http://cs231n.github.io/classification) [[linear classification notes\]](http://cs231n.github.io/linear-classify) |
| Lecture 3      | 18-08-12 | **Loss Functions and Optimization**  Linear classification II Higher-level representations, image features Optimization, stochastic gradient descent | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf) [[video\]](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [[linear classification notes\]](http://cs231n.github.io/linear-classify) [[optimization notes\]](http://cs231n.github.io/optimization-1) |
| Lecture 4      | 18-08-12 | **Introduction to Neural Networks**  Backpropagation Multi-layer Perceptrons The neural viewpoint | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf) [[video\]](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [[backprop notes\]](http://cs231n.github.io/optimization-2) [[linear backprop example\]](http://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf) [[derivatives notes\]](http://cs231n.stanford.edu/2017/handouts/derivatives.pdf) (optional)  [[Efficient BackProp\]](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (optional) related: [[1\]](http://colah.github.io/posts/2015-08-Backprop/), [[2\]](http://neuralnetworksanddeeplearning.com/chap2.html), [[3\]](https://www.youtube.com/watch?v=q0pm3BrIUFo) (optional) |
| Lecture 5      | 18-08-26 | **Convolutional Neural Networks**  History  Convolution and pooling  ConvNets outside vision | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf) [[video\]](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [ConvNet notes](http://cs231n.github.io/convolutional-networks/) |
| Lecture 6      | 18-08-26 | **Training Neural Networks, part I**  Activation functions, initialization, dropout, batch normalization | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) [[video\]](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [Neural Nets notes 1](http://cs231n.github.io/neural-networks-1/) [Neural Nets notes 2](http://cs231n.github.io/neural-networks-2/) [Neural Nets notes 3](http://cs231n.github.io/neural-networks-3/) tips/tricks: [[1\]](http://research.microsoft.com/pubs/192769/tricks-2012.pdf), [[2\]](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), [[3\]](http://arxiv.org/pdf/1206.5533v2.pdf) (optional)  [Deep Learning [Nature\]](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html) (optional) |
| A1 Due         |          | **Assignment #1 due**  kNN, SVM, SoftMax, two-layer network  | [[Assignment #1\]](http://cs231n.github.io/assignments2017/assignment1/) |
| Lecture 7      | 18-09-02 | **Training Neural Networks, part II**  Update rules, ensembles, data augmentation, transfer learning | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf) [[video\]](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [Neural Nets notes 3](http://cs231n.github.io/neural-networks-3/) |
| Lecture 8      | 18-09-02 | **Deep Learning Software**  Caffe, Torch, Theano, TensorFlow, Keras, PyTorch, etc | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture8.pdf) [[video\]](https://www.youtube.com/watch?v=6SlgtELqOWc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 9      | 18-09-09 | **CNN Architectures**  AlexNet, VGG, GoogLeNet, ResNet, etc  | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf) [[video\]](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [VGGNet](https://arxiv.org/abs/1409.1556), [GoogLeNet](https://arxiv.org/abs/1409.4842), [ResNet](https://arxiv.org/abs/1512.03385) |
| Lecture 10     | 18-09-09 | **Recurrent Neural Networks**  RNN, LSTM, GRU  Language modeling  Image captioning, visual question answering  Soft attention | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf) [[video\]](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [DL book RNN chapter](http://www.deeplearningbook.org/contents/rnn.html) (optional) [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086), [char-rnn](https://github.com/karpathy/char-rnn), [neuraltalk2](https://github.com/karpathy/neuraltalk2) |
| A2 Due         |          | **Assignment #2 due**  Neural networks, ConvNets             | [[Assignment #2\]](http://cs231n.github.io/assignments2017/assignment2/) |
| Lecture 11     | 18-09-16 | **Detection and Segmentation**  Semantic segmentation  Object detection  Instance segmentation | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf) [[video\]](https://www.youtube.com/watch?v=nDPWywWRIRo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 12     | 18-09-16 | **Visualizing and Understanding**  Feature visualization and inversion  Adversarial examples  DeepDream and style transfer | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf) [[video\]](https://www.youtube.com/watch?v=6wcs6szJWMY&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [DeepDream](https://github.com/google/deepdream) [neural-style](https://github.com/jcjohnson/neural-style) [fast-neural-style](https://github.com/jcjohnson/fast-neural-style) |
| Lecture 13     | 18-09-23 | **Generative Models**  PixelRNN/CNN  Variational Autoencoders  Generative Adversarial Networks | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf) [[video\]](https://www.youtube.com/watch?v=5WoItGTWV54&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 14     | 18-09-23 | **Deep Reinforcement Learning**  Policy gradients, hard attention  Q-Learning, Actor-Critic | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf) [[video\]](https://www.youtube.com/watch?v=lvoHnicueoE&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| A3 Due         |          | **Assignment #3 due**                                        | [[Assignment #3\]](http://cs231n.github.io/assignments2017/assignment3/) |
| Lecture 15, 16 | 18-09-30 | Student spotlight talks, conclusions                         | [slides]                                                     |



## 2. 모두를 위한 딥러닝

#### 계획 

- 강의(선택)
- 코드 작성 및 주석 달아올 것



#### 커리큘럼

8/5 : ~ Softmax

8/12 : Back Propagation

8/26 : Neural Network

9/2 : CNN

9/9 : RNN

