---
title: "Optimization with Python 03 - Linear Programming(LP)"
excerpt: "Optimization with Python 03 - Linear Programming(LP)"
date: 2022-02-22 00:00:00 +0900
header:
  overlay_image: /assets/images/unsplash-thomas-t-math.jpg
  overlay_filter: 0.5
  caption: "Photo by [**Thomas T**](https://unsplash.com/@pyssling240) on [**Unsplash**](https://unsplash.com/)"
categories:
  - Optimization
mathjax: "true"
---
**Notice:** 본 글은 Udemy의 Optimization with Python: Solve Operations Research Problems을 학습하며 정리한 글입니다.
{: .notice--info}

**Warning Notice:** 본 강의는 파이썬을 이용하여 최적화 문제를 푸는 것에 집중합니다. 파이썬 언어의 기초는 다루지 않으므로 주의하십시오.
{: .notice--warning}

# Introduction
 아래의 문제를 풀어보자.

![solution]({{site.baseurl}}/assets/images/2022-02-21-optimization-with-python-solution.png)

- Objecttive
  - $\max x+y$
- Constraints
  - $-x+2y\le8$
  - $2x+y\le14$
  - $2x-y\le10$
  - $0 \le x \le 10$
  - $0 \le y \le 10$

## Solver vs Framework
1. 문제 이해
2. 문제 모델링
3. 프로그래밍 언어(Framework)
4. 문제 풀이(Solver)
5. 결과

## LP: Ortools

```sh
pip install -U ortools
```

Ortools 실습

```python
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver('GLOP')

# Variable 선언, 0: Lower Bound, 10: Upper Bound
x = solver.NumVar(0, 10, 'x') # Constraint 0 <= x <= 10
y = solver.NumVar(0, 10, 'y')

solver.Add(-x+2*y<=8)
solver.Add(2*x+y<=14)
solver.Add(2*x-y<=10)
```

<ortools.linear_solver.pywraplp.Constraint; proxy of <Swig Object of type 'operations_research::MPConstraint *' at 0x00000154435CDBA0> >

```python
solver.Maximize(x+y)

results = solver.Solve()

if results == pywraplp.Solver.OPTIMAL:
    print('Optimal Found')
```
Optimal Found

```python
print('x:', x.solution_value())
print('y:', y.solution_value())
```
x: 4.0  
y: 6.0
