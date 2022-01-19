---
title: "파이썬 가상 환경"
date: 2020-04-20 21:00:00 +0900
categories:
  - Python
---

## Overview of Python Virtual Environments

### 파이썬 가상 환경이란?

- 기본적으로 모든 파이썬 패키지들은 하나의 디렉토리에 설치된다.
- 이것은 여러 파이썬 프로젝트를 작업할 때 의존성 문제가 발생할 수 있다.
- 예를 들어, 프로젝트 A가 어떤 라이브러리의 버전 1을 사용하는데, 프로젝트 B는 버전 2를 사용한다고 생각해보자.
- 파이썬 런타임에 동일한 라이브러리의 두 버젼을 둘다 사용할 수 없기 때문에 문제가 발생한다.
- 가상 환경은 이 문제를 프로젝트별 분리된 환경을 제공함으로써 이 문제를 해결한다.
- 즉, 가상 환경의 파이썬은 파이썬이 실행될 때 찾아지고 실행된다. 그리고 해당 가상 환경에 설치된 패키지들을 실행한다.

### 파이썬3 가상환경 생성하기

- 파이썬3는 venv라는 빌트인 모듈이 있다.
- 파이썬2에서 사용하던 virtualenv 모듈을 파이썬3에서 사용할 수는 있지만, 파이썬3에서는 venv를 사용하는 것을 추천한다.

```bash
# 동작이 제대로 안 될 시, python 대신 python의 full path를 넣어준다.(Ex. C:\Path\to\python.exe -m venv my_env)
python3 -m venv <Virtual Environment Name>
```

### 파이썬3 가상환경 활성화하기
```bash
# Linux
source <Virtual Environment Name>/bin/activate
```

### Windows에 Python 가상 환경 세팅하기

- Windows용 bash shell을 이용하기(i.e. cygwin.com)
- 혹은, 튜토리얼의 Windows용 command line 참조하기(python.org)
