---
title: "tmux 사용법"
excerpt: "tmux 사용법"
tagline: "tmux 사용법"
header:
  overlay_image: /assets/images/2021-12-08-terminal.jpg
  overlay_filter: 0.5
  caption: "Photo creadit: [**Unsplash**](https://unsplash.com)"
date: 2021-12-08 23:00:00 +0900
categories:
  - Linux
---
# tmux 사용법
나 같은 경우에는 주로 원격 접속 시 세션 관리를 위해 tmux 프로그램을 사용한다.  
putty나 terminus로 원격 접속 시 프로그램이 꺼지면 세션 정보도 사라져서, 새로 접속할 때마다 초기화되는 불편함이 있다.
tmux를 사용하면 기존에 사용하던 세션을 불러올 수도 있고, 또한 동시에 여러가지 작업을 할 때 여러 세션을 사용하면 편하다.

설치 명령어는 `sudo apt install tmux`로 설치할 수 있다.

- 세션 확인: `tmux list-session`, `tmux ls`
- 세션 생성(세션명 자동): `tmux`, `tmux new`
- 세션 생성(세션명): `tmux new-session -s 세션명`, `tmux new -s 세션명`
- 세션 이름 변경: `<Ctrl + b> $`
- 세션 Detach(세션을 살려둔채 벗어남): `<Ctrl + b> d`
- 세션 Attach(기존 세션으로 복귀): `tmux a -t 세션명`, `tmux attach -t 세션명`
- 세션 종료: `tmux kill-session -t 세션명`, `(해당 세션에서)exit`
- History 스크롤하기: `<Ctrl + b + [>`로 스크롤 모드 진입 후 방향키, Page Up/Down으로 스크롤, `q`로 종료

- 세션 내 새 Window 생성: `<Ctrl + b> c`
- Window간 이동: `<Ctrl + b> [0~9 번호]`, `<Ctrl + b> n(next), p(prev), l(Last), w(select), f(find)`
- Window명 변경: `<Ctrl + b> ,`
- Window 종료: `<Ctrl + b> &`, `<Ctrl + d>`

- Pane split(가로): `<Ctrl + b> %`
- Pane split(세로): `<Ctrl + b> "`
- Pane 이동: `<Ctrl + b> 방향키`
