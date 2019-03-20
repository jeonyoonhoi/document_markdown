# Lecture 2. Image Classifcatioon

### tips

- Python + Numpy
- Google Cloud Tutorial



## Image Classifacation

A core task in Computer Vision



### 문제점 : semantic



### Challenges : 

* Viewpoint variation, 
* Illumination,
* Deformation, 
* Occlusion, 
* Background cluster, 
* Intraclass variation



### Data Driven Approach

1. Collect a dataset of images and labels
2. Use Machine Learning to train a classifier
3. Evaluate the classifier on new images



### 1st classifier :  Nearest Neighbor

```python 
def train(images, labels):
    #machine learning!
    return model
```

Memorize all data and labels

``` python
def predict(model, test_images):
	#Use model to predict labels
    return test_labels
```

Predict the label of the most similar training image



- train : making a model
- predict : with the model, making a prediction



### Example Dataset : CIFAR10



이전에 보면 L1 디스턴스는 비교 이미지, 나머지가 테스트 이미지



### Distance Metic to compare images

정말 간단하면서도 바보같은 방법인가!

```
test image - training immage = pixel-wise absolute value differences
```

각 픽셀값을 뺀 다음 전부를 더해서 유사도를 측정한다?



prediction은 빠르지만 trainign이 느리다. 

O(N), O(1)



### K-nearest Neighbors

Instead of copying label from mearest neighbor, take majority vote from K closest points



![1533362560348](C:\Users\YOONHOI\Documents\cs231n\img\1533362560348.png)



choosing K, Matrics

L1 ,L2의 특징이 있고, 문제에 따라 다르게 사용하지만

정답은 둘 다 해보고 더 나은 것을 쓴ㄴ 것. 



### Hyperparameters

* What is the best value of k to use?
* What is the best distance to use?

These 



### Setting Hyperparameters

![1533363453030](C:\Users\YOONHOI\Documents\cs231n\img\1533363453030.png)

![1533363526296](C:\Users\YOONHOI\Documents\cs231n\img\1533363526296.png)







Example of 5-hold cross-validation for the value of k.



### k-Nearest N

### eighbor on images never used.

- 왜곡된 이미지
- curse of dimensionality



K-Nearest Neighbors : Summary



In Image Classification we start with a training set of images and labels, and must predict labels on the test set

The K-Nearest Neighborss classifier preicts labels based on nearest traininng examples.

Distance metric and K are hyperparameters





## Linear Classifier



