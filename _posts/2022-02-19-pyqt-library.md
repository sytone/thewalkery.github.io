---
title: "PyQt를 이용한 도서관 관리 시스템 개발"
excerpt: "PyQt를 이용한 도서관 관리 시스템 개발"
date: 2022-02-19 22:00:00 +0900
header:
  overlay_image: /assets/images/unsplash-emile-perron.jpg
  overlay_filter: 0.5
  caption: "Photo by [**Emile Perron**](https://unsplash.com/@emilep) on [**Unsplash**](https://unsplash.com/)"
categories:
  - PyQt
---
**Notice:** 본 글은 Udemy의 Build Library Management System \| Python & PyQt5을 학습하며 정리한 글입니다.
{: .notice--info}

# 01. Couse Introduction
- Design With Qt Designer
- Design MySQL Database
- Connecting UI & DB To Python
- Adding App Themes
- Adding & Editing Users
- Export Data To Excel Files

# 02. Tools Setup
- Python
- MySQL Community Server
- MySQL Workbench
- PyCharm

# 03. Project Structure
library system:  
- add new book
- editing book
- deleting book
- categories
- search
- user, login, signup
- settings [categories, author, publisher]
- day to day operations
- reports [excel files]

books:
- title
- code
- description
- category
- price
- author
- publisher
  
users:  
- username
- password
- email address

# 04. 시작하기
- 프로젝트 폴더 생성하기
- 프로젝트 폴더 내에 가상 환경 생성 및 활성화
- PyQt5 설치
- MySQL, MySQL Workbench 설치

```sh
# 프로젝트 폴더에서 cmd 실행
# 가상 환경 생성하기(가상 환경 이름: .venv)
<Python 실행 파일> -m venv .venv

# 가상 환경 활성화하기
.venv\Scripts\activate

# PyQt5 설치
pip install pyqt5

# PyQt5 툴 설치
pip install pyqt5-tools
```

- PyQt5 관련 툴 실행법

```sh
# QtDesigner 실행
pyqt5-tools designer

# rcc (*.qrc -> *.py)
pyrcc5 icons.qrc -o icons_rc.py

# uic (*.ui -> *.py)
pyuic5 library.ui -o library.py

# Build an executable file(Output directory: build/)
python setup.py build
```

# 05. 결과
 ![screenshot]({{site.baseurl}}/assets/images/2022-02-19-pyqt-screenshot.png)  

모든 기능을 구현하지는 않았고, 주요 Feature(signal/slot, ui, resource, freeze) 일부만 적용하였다. MySQL 및 Excel Write 기능은 미완성 상태이다.

내 코드는 [github repository](https://github.com/thewalkery/pyqt5-practice-library) 참조

강사의 전체 코드는 [강사의 github repository](https://github.com/Pythondeveloper6/Build-Library-Management-System-Python-PyQt5) 참조


