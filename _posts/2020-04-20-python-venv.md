---
title: "파이썬 가상 환경"
date: 2020-04-20 21:00:00 +0900
categories:
  - Programming
tags:
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

### 가상 환경 설치

```bash
# 설치
pip install virtualenv
# 새로운 환경 만들기
virtualenv <NameOfVirtualEnv>
# 환경 활성화 하기: 해당 가상환경의 bin directory의 activate 스크립트를 소싱한다.
source ./<NameOfVirtualEnv>/bin/activate
# Windows의 경우 cmd창을 띄워서 <NameOfVirtualEnv>\Scripts\activate.bat 실행하면 된다.
# 비활성화하기
deactivate
# directory를 삭제하면 virtualenv가 삭제된다.
```

### 파이썬3에 가상환경 설치하기

- 파이썬3는 venv라는 빌트인 모듈이 있다.
- Virtualenv도 파이썬3에서 사용할 수는 있지만, 파이썬3에서는 venv를 추천한다.
- venv는 virtualenv와 사용되는 커멘드만 다를 뿐 사용법은 동일하다.

```bash
# 동작이 제대로 안 될 시, python 대신 python의 full path를 넣어준다.(Ex. C:\Path\to\python.exe -m venv my_env)
python3 -m venv <VirtualEnvironmentName>
```

### Windows에 Python 가상 환경 세팅하기

- Windows용 bash shell을 이용하기(i.e. cygwin.com)
- 혹은, 튜토리얼의 Windows용 command line 참조하기(python.org)
