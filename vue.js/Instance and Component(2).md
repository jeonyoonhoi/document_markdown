# 화면을 개발하기 위한 필수 단위 : 인스턴스 & 컴포넌트

## 03-2 뷰 컴포넌트

### 컴포넌트란?



컴포넌트(Component)란 조합화여 화면을 구성할 수 있는 블록을 의미



뷰에서는 웹화면을 구성할 떄 흔히 사용하는 내ㅐ비게이션 바, 테이블, 리스트, 인풋박스 등과 같은 화면 구성요소들을 잘게 쪼개어 컴퓨넌트로 관리한다. 



### 컴포넌트 등록하기

컴포넌트를 등록하는 방법으로는 전역과 지역의 두 가지가 있다. 

- 지역 컴포넌트는 특정 인스턴스에서만 유효한 범위를 갖고
- 전역 컴포넌트는 여러 인스턴스에서ㅓ 공통으로 사요ㅛㅇ할 수 있다. 



#### 전역 컴포넌트 등록

전역 컴포넌트는 뷰 라이브러리를 로딩하고 나면 접근 가능한 Vue 변수를 이용하여 등록. 

전역 컴포넌트를 모든 인스턴스에 등록하려면 Vue생성자에서 .component()를 호출하여 수행하면 된다. 

``` 전역 컴포넌트 등록 형식
Vue.component('컴포넌트 이름', {
    //컴포넌트 내용
})
```

- 컴포넌트 이름 :  template속성에서 사용할 HTML 사용자 정의 태그 이름
- 컴포넌트 내용 : 태그가 실제 화면의 HTML요소로 변환될 때 표시 될 속성들을 정의 
  - template, data, methods 등 인스턴스 옵션 속성을 정의할 수 있음



```html
<html>
    <head>
        <title>Vue Component registration</title>
    </head>
    <body>
      <div id = 'app'>
        <button> 컴포넌트 등록 </button>
        <my-Component></my-Component>    <!--전역 컴포넌트 표시-->
      </div>

      <script src = 'https://cdn.jsdelivr.net/npm/vue@2.5.2/dist/vue.js'></script>
      <script>
        Vue.component('my-component', {
          template : '<div> 전역 컴포넌트가 등록되었습니다!</div>'
        })

        new Vue({
          el : '#app'
        });
      </script>
    </body>
</html>

```



![1533579195368](C:\Users\YOONHOI\AppData\Local\Temp\1533579195368.png)



```컴포넌트 태그 추가
<my-component></my-component>	
```

라고 적힌 부분이 실제로는

```
<div>전역 컴포넌트가 등록되었습니다 </div>
```

실제로는 이렇게 화면에 그려진다. 



#### 지역 컴포넌트 등록

``` 지역 컴포넌트 등록 형식
new Vue({
    component : { 
    '컴포넌트 이름' :컴포넌트 내용}
})
```

컴포넌트 이름은 전역 컴포넌트와 마찬가지로 HTML 에 등록할 사용자 정의 태그를 의미하고, 컴포넌트 내용은 컴포넌트 태그가 실제 화면 요소로 변환될 때의 내용을 의미한다. 



``` javascript
<html>
    <head>
        <title>Vue local Component registration</title>
    </head>
    <body>
      <div id = 'app'>
        <button> 지역 컴포넌트 등록 </button>
        <my-local-component></my-local-component>    <!--전역 컴포넌트 표시-->
      </div>

      <script src = 'https://cdn.jsdelivr.net/npm/vue@2.5.2/dist/vue.js'></script>
      <script>
        var cmp = {
          //컴포넌트 내용
          template : '<div>지역 컴포넌트가 등록되었습니다. </div>'
        }
		

        new Vue({
          el : '#app'
          components : {
            'my-local-component' : cmp
          }

        });
      </script>
    </body>
</html>

```



#### 지역 컴포넌트와 전역 컴포넌트의 차이

