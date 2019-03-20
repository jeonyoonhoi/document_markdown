### Linear Regression



#### TF 의 텐서, 상수, 변수, 플레이스 홀더 

- 텐서플로우에는 세 가지의 핵슴 데이터 구조인 다음 세 가지가 존재한다. 
  - Constant 상수
  - Variable 변수
  - Placeholder 플레이스 홀더



차이점 : ```tf.Variable```을 사용하면 선언할 때 초기 값을 제공해야 한다. ``` tf.placeholder```를 사용하면 초기 값을 제공 할 필요가 없으므로 ```Session.run```의 ```feed_dict``` 인수를 사용하여 런타임에 지정할 수 있다. 



`Variables(tf.varibales(SomeValue)):`

- 값은 train을 통해 얻어질 수 있음
- 초기 값 필요(종종 무작위)
- 변수는 현재 값 (우리가 할당 한 값)을 출력하는 상태 저장 노드

`Placeholders(tf.placeholders(dtype, shape)):`

- 데이터를 위해 할당 된 저장소
- 초기 값은 필요하지 않지만 설정할 순 있응ㅁ
- 실행시에 값이 제공되는 노드
-  예를 들어, 입력, 레이블 등



#### Hypoothesis and cost function


$$
H(x) = Wx+b
$$

$$
cost(W,b) = 1/m 
$$

