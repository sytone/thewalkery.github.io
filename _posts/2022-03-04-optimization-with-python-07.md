---
title: "Optimization with Python 07 - Constraint Programming (CP)"
excerpt: "Optimization with Python 07 - Constraint Programming (CP)"
date: 2022-03-04 10:00:00 +0900
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

# Constraint Programming (CP)
- 문제의 Objective function 보다 Constraint가 중요할 때 사용
- Constraint가 Tight할 때
- Objective function이 서로 비슷비슷한 수준일 때

# Ortools

Reference: https://developers.google.com/optimization/cp/cp_example

Objective
- $\max 2x + 2y + 3z$

Constraints
- $x + \frac{7}{2} y + \frac{3}{2}z \le 25$
- $3x - 5y + 7z \le 45$
- $5x + 2y - 6z \le 37$
- $x, y, z \ge 0$
- $x, y, z$ integers

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

x = model.NewIntVar(0, 1000, 'x')
y = model.NewIntVar(0, 1000, 'y')
z = model.NewIntVar(0, 1000, 'z')

model.Add(2*x+7*y+3*z<=50)
model.Add(3*x-5*y+7*z<=45)
model.Add(5*x+2*y-6*z<=37)

model.Maximize(2*x+2*y+3*z)

solver = cp_model.CpSolver()
status = solver.Solve(model)

print('Status = ', solver.StatusName(status))
print('FO =', solver.ObjectiveValue())
print('x =', solver.Value(x))
print('y =', solver.Value(y))
print('z =', solver.Value(z))
```

<div class="no-highlight" markdown="1">

```
    Status =  OPTIMAL
    FO = 35.0
    x = 7
    y = 3
    z = 5
```

</div>

# Resources

- [Constraint Programming]({{site.baseurl}}/assets/resources/2022-03-04-optimization-with-python-07-CP.pdf)
