---
title: "Qt Project - Minimal Timer 01"
excerpt: "Qt Project - Minimal Timer 01"
date: 2022-03-06 10:00:00 +0900
header:
  overlay_image: /assets/images/unsplash-emile-perron.jpg
  overlay_filter: 0.5
  caption: "Photo by [**Emile Perron**](https://unsplash.com/@emilep) on [**Unsplash**](https://unsplash.com/)"
categories:
  - Qt Project - Minimal Timer
mathjax: "true"
---

# 프로젝트 개요

- 집중을 위한 타이머 프로그램 개발하자.
- 남은 시간을 시각적으로 보여주는 타이머

![Timer]({{site.baseurl}}/assets/images/2022-03-06-time-timer.gif)


# 프로젝트 스펙

- 마우스 클릭/드래그를 통해 타이머 지정할 수 있다.
- 빨간색 영역이 남은 시간을 뜻하며, 시간이 지날수록 빨간색 영역이 줄어든다.
- 남은 시간이 0이 되면 사용자에게 알린다(Pop Up/Alarm 소리 등).
- 테마 설정 기능

## 참고용 영상 - Time Timer

{% include video id="DxpFMMq763Q" provider="youtube" %}


# 화면 설계
- 미니멀하게 가져가자.



# 프로젝트 시작하기

## Github 프로젝트 생성
- 프로젝트 관리를 위해 Github에 Repository 생성
- [Github 프로젝트 링크](https://github.com/thewalkery/minimal-timer)

## 로컬 프로젝트 생성
- `git clone https://github.com/thewalkery/minimal-timer.git`

## 가상 환경 셋업
- `python -m venv .venv`
- `.venv\Scripts\activate`
- `pip install pyqt5`

## 참고 자료
- [Python and PyQt: Building a GUI Desktop Calculator](https://realpython.com/python-pyqt-gui-calculator/)