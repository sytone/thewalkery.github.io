---
title: "Fluent Python 01 - Python Data Model"
excerpt: "Fluent Python 01 - Python Data Model"
date: 2022-03-20 00:00:00 +0900
header:
  overlay_image: /assets/images/unsplash-emile-perron.jpg
  overlay_filter: 0.5
  caption: "Photo by [**Emile Perron**](https://unsplash.com/@emilep) on [**Unsplash**](https://unsplash.com/)"
categories:
  - Fluent Python
---

**Notice:** 본 글은 책 [『전문가를 위한 파이썬<sup>Fluent Python</sup>』](https://books.google.co.kr/books?id=NJpIDwAAQBAJ&hl=ko&source=gbs_navlinks_s)을 학습하며 정리한 글입니다. 전체 소스 코드는 [Fluent Python github 레포지토리](https://github.com/fluentpython/example-code)에서 확인할 수 있습니다.
{: .notice--info}

# Chapter 1. 파이썬 데이터 모델

> 언어 설계 미학에 대한 귀도의 감각은 놀라울 정도다.<br>아무도 시용하지 않을 이론적으로 아름다운 언어를 설계할 능력이 있는 훌륭한 언어 설계자를 많이 만났지만,<br>귀도는 이론적으로는 약간 덜 아름답지만 그렇기 때문에 프로그래밍하기 즐거운 언어를 설계할 수 있는 유례없는 능력자 중 한 사람이다.<br>**_ 짐 허구닌<sup>Jim Hugunin</sup><br>Jython의 창시자, AspectJ의 공동 설계자, .Net DLR 아키텍트**

파이썬의 장점 중 하나는 일관성이다. 파이썬을 배우기 전에 다른 객체지향<sup>object-oriented</sup> 언어를 배웠다면 `collection.len()` 대신 `len(collection)`을 사용하는 점을 이상하게 생각할 수도 있다. 이런 괴상함은 빙산의 일각이며, 적절히 이해하면 소위 **파이썬스러움**<sup>pythonic</sup>이라고 부르는 것의 핵심을 간파할 수 있다. 이 빙산을 '파이썬 데이터 모델'이라고 하며, 파이썬 데이터 모델이 제공하는 API를 이용해서 여러분 고유의 객체를 정의하면 대부분의 파이썬 상용구를 적용할 수 있다.  
파이썬 데이터 모델은 일종의 프레임워크로서, 언어 자체의 구성단위에 대한 인터페이스를 공식적으로 제공한다.  
프레임워크를 이용해서 코딩할 때는 프레임워크에 의해 호출되는 메서드를 구현하는 데 많은 시간을 소비한다. 파이썬 데이터 모델을 사용할 때도 마찬가지다. 예를 들어 `obj[key]` 형태의 구문은 `__getitem__()` 특별 메서드가 지원한다.  

## 1.1 파이썬 카드 한 벌

```python
import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
```

이 코드에서는 `collections.namedtuple()`을 이용해서 개별 카드를 나타낸 클래스를 구현한다. 이 클래스를 이용하면 다음과 같이 카드 한 장을 만들 수 있다.

```python
>>> beer_card = Card('7', 'diamonds')
>>> beer_card
Card(rank='7', suit='diamonds')
```

이 코드의 핵심은 `FrenchDeck` 클래스이다. 이 코드는 간단하지만 아주 많은 기능을 구현한다. 먼저 일반적인 파이썬 컬렉션과 마찬가지로 `len()` 함수를 통해 자신이 갖고 있는 카드의 수를 반환한다. 또한, index를 사용하여 특정 카드를 가져올 수 있다.

```python
>>> deck = FrenchDeck()
>>> len(deck)
>>> deck[0]
Card(rank='2' , suit='spades' )
>>> deck[-1]
Card(rank='A' , suit='hearts' )
```

<span class="custom-highlight">**임의의 카드를 골라내는 메서드를 정의해야할까? 그럴 필요 없다.**</span> 파이썬은 시퀀스에서 항목을 무작위로 골라내는 `random.choice()`라는 메서드를 제공한다.

```python
>>> from random import choice
>>> choice(deck)
Card(rank='3', suit='hearts')
>>> choice(deck)
Card(rank='K', suit='spades')
>>> choice(deck)
Card(rank='2', suit='clubs')
```

여기서 우리는 특별 메서드를 통해 파이썬 데이터 모델을 사용할 때의 두 가지 장점을 알게 되었다.
- 사용자가 표준 연산을 수행하기 위해 클래스 자체에서 구현한 임의 메서드명을 암기할 필요가 없다. 예를 들어, 항목 수를 알기 위해서 `size()`를 사용해야 하나? 아니면 `length()`?
- 파이썬 표준 라이브러리에서 제공하는 풍부한 기능을 별도로 구현할 필요 없이 바로 사용할 수 있다(`random.choice()` 함수처럼).

## 1.2 특별 메서드는 어떻게 사용되나?
특별 메서드에 대해 알아두어야 할 점은 특별 메서드는 사용자가 아니라 파이썬 인터프리터가 호출하기 위한 것이라는 점이다. 사용자 소스 코드에서는 `my_object.__len__()`으로 직접 호출하지 않고, `len(my_object)` 형태로 호출한다. 만약 `my_object`가 사용자 정의 클래스의 객체면 파이썬은 해당 클래스에 구현된 `__len__()` 객체 메서드를 호출한다.  
그러나 `list`, `str` 등과 같은 내장 자료형의 경우 내장 구현되어있는 최적화된 방법을 사용한다. 일반적으로 사용자 코드에서는 특별 메서드를 직접 호출하는 경우는 그리 많지 않다.

### 1.2.1 수치형 흉내 내기
덧셈(+)과 같은 연산자에 사용자 정의 객체가 응답할 수 있게 해주는 몇몇 특별 메서드가 있다. 수학이나 물리학에서 사용되는 2차원 유클리드 벡터를 나타내는 클래스를 구현해보자.  
먼저 이런 클래스에 대한 API 설계하자.
```python
# 덧셈
>>> v1 = Vector(2, 4)
>>> v2 = Vector(2, 1)
>>> v1 + v2
Vector(4, 5)

# 절대값
>>> v = Vector(3, 4)
>>> abs(v)
5.0

# 스칼라곱
>>> v * 3
Vector(9, 12)
>> abs(v * 3)
15.0
```
다음은 특별 메서드를 이용해서 위에서 설명한 연산을 구현하는 Vector 클래스다.
```python
from math import hypot

class Vector:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
```

### 1.2.2 문자열 표현
`__repr__()` 특별 메서드는 객체를 문자열로 표현하기 위해 `repr()` 내장 메서드에 의해 호출된다. 만일 `__repr__()` 메서드를 구현하지 않으면 Vector 객체는 콘솔에 `<Vector object at 0x10e100070>`과 같은 형태로 출력된다.  
`__repr__()` 메서드가 반환한 문자열은 명확해야 하며, 가능하면 표현된 객체를 재생성하는 데 필요한 소스 코드와 일치해야 한다. 반면, `__str__()`은 메서드는 `str()` 생성자에 의해 호출되며 `print()` 함수에서 암묵적으로 사용된다. 두 특별 메서드 중 하나만 구현한다면 `__repr__()` 메서드를 구현하자. 파이썬 인터프리터는 `__str__()` 메서드가 구현되어 있지 않을 때 `__repr__()` 메서드를 사용한다.

### 1.2.3 산술 연산자
위 예제에서 `__add__()`와 `__mul__()`의 기본 사용법을 보여주기 위해 덧셈(+)과 곱셈(*) 연산자를 구현한다. 두 메서드 모두 Vector 객체를 새로 만들어서 반환하며 두 개의 피연산자는 변경하지 않는다.

**주의:** 위 예제 코드는 Vector에 숫자를 곱할 수는 있지만, 숫자에 Vector를 곱할 수는 없다. 이 문제는 추후 `__rmul__()` 메서드를 이용해서 수정한다.
{: .notice--warning}

### 1.2.4 사용자 정의형의 불리언 값
x가 True인지 False인지 판단하기 위해 파이썬은 `bool(x)`를 적용하며, 이 함수는 항상 `True`나 `False`를 반환한다.  
`__bool__()`이나 `__len__()`을 구현하지 않은 경우, 기본적으로 사용자 정의 클래스의 책체는 `True`로 간주된다. 기본적으로 `bool(x)`는 `x.__bool__()`을 호출한 결과를 이용한다. 만약 이 bool 특별 메서드가 구현되어 있지 않으면 파이썬은 `x.__len__()`을 호출하여 0을 반환하면 False, 그렇지 않으면 True로 판단한다.  
우리가 구현한 `__bool__()`은 벡터의 크기가 0이면 False, 0이 아니면 True를 반환한다.

## 1.3 특별 메서드 개요
파이썬 언어 참조 문서의 [데이터 모델 장](https://docs.python.org/3/reference/datamodel.html)에서는 83개 특별 메서드가 소개되어 있다.

|Group|Methods|
|-----|-------|
|문자열/바이트 표현|\_\_repr\_\_, \_\_str\_\_, \_\_format\_\_, \_\_bytes\_\_|
|숫자 변환| \_\_abs\_\_, \_\_bool\_\_, \_\_complex\_\_, \_\_int\_\_, \_\_float\_\_, \_\_hash\_\_, \_\_index\_\_ |
|컬렉션 에뮬레이션| \_\_len\_\_, \_\_getitem\_\_, \_\_setitem\_\_, \_\_delitem\_\_, \_\_contains\_\_ |
|반복| \_\_iter\_\_, \_\_reversed\_\_, \_\_next\_\_ |
|콜러블 에뮬레이션| \_\_call\_\_ |
|콘텍스트 관리| \_\_enter\_\_, \_\_exit\_\_ |
|객체 생성 및 소멸| \_\_new\_\_, \_\_init\_\_, \_\_del\_\_ |
|속성 관리| \_\_getattr\_\_, \_\_getattribute\_\_, \_\_setattr\_\_, \_\_delattr\_\_, \_\_dir\_\_ |
|속성 디스크립터| \_\_get\_\_, \_\_set\_\_, \_\_delete\_\_ |
|클래스 서비스| \_\_prepare\_\_, \_\_instancecheck\_\_, \_\_subclasscheck\_\_ |

|Group| Methods & Operators |
|-----|---------------------|
|단항 수치 연산자| \_\_neg\_\_ -, \_\_pos\_\_ +, \_\_abs\_\_ abs()|
|비교 연산자| \_\_lt\_\_ <, \_\_le\_\_ <=, \_\_eq\_\_ ==, \_\_ne\_\_ !=, \_\_gt\_\_ >, \_\_ge\_\_ >= |
|산술 연산자| \_\_add\_\_ +, \_\_sub\_\_ -, \_\_mul\_\_ *, \_\_truediv\_\_ /, \_\_floordiv\_\_ //, \_\_mod\_\_ %, \_\_divmod\_\_ divmode(), \_\_pow\_\_ ** 혹은 pow(), \_\_round\_\_ round() |
|역순 산술 연산자| \_\_radd\_\_, \_\_rsub\_\_, \_\_rmul\_\_, \_\_rtruediv\_\_, \_\_rfloordiv\_\_, \_\_rmod\_\_, \_\_rdivmod\_\_, \_\_rpow\_\_|
|복합 할당 산술 연산자| \_\_iadd\_\_, \_\_isub\_\_, \_\_imul\_\_, \_\_itruediv\_\_, \_\_ifloordiv\_\_, \_\_imod\_\_, \_\_idivmod\_\_, \_\_ipow\_\_|
|비트 연산자| \_\_invert\_\_ ~, \_\_lshift\_\_ <<, \_\_rshift\_\_ >>, \_\_and\_\_ &, \_\_or\_\_ \|, \_\_xor\_\_ ^ |
|역순 비트 연산자| \_\_rlshift\_\_, \_\_rrshift\_\_, \_\_rand\_\_, \_\_rxor\_\_, \_\_ror\_\_ |
|복합 할당 비트 연산자| \_\_ilshift\_\_, \_\_irshift\_\_, \_\_iand\_\_, \_\_ixor\_\_, \_\_ior\_\_ |

- 역순 연산자: 피연산자의 순서가 바뀌었을 경우(a * b 대신 b * a)
- 복합 할당 연산자: 연산과 변수 할당을 줄여서 표현하는 경우(a = a + b를 a += b로)