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

## Linear Programming: Ortools

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

<div class="no-highlight" markdown="1">

```
<ortools.linear_solver.pywraplp.Constraint; proxy of <Swig Object of type 'operations_research::MPConstraint *' at 0x00000154435CDBA0> >
```

</div>

```python
solver.Maximize(x+y)

results = solver.Solve()

if results == pywraplp.Solver.OPTIMAL:
    print('Optimal Found')
```

<div class="no-highlight" markdown="1">

```
Optimal Found
```

</div>

```python
print('x:', x.solution_value())
print('y:', y.solution_value())
```

<div class="no-highlight" markdown="1">

```
x: 4.0  
y: 6.0
```

</div>

## Linear Programming: SCIP
- www.scipopt.org에서 다운로드/설치 필요(+ 환경변수 설정)  
- package PYSCIPOPT 설치
- Non-linear 문제에서도 동작한다.

```python
from pyscipopt import Model

model = Model('exemplo')

x = model.addVar('x')
y = model.addVar('y')

model.setObjective(x+y, sense='maximize')
model.addCons(-x+2*y<=8)
model.addCons(2*x+y<=14)
model.addCons(2*x-y<=10)

model.optimize()

sol = model.getBestSol()

print('x=', sol[x])
print('y=', sol[y])
```

<div class="no-highlight" markdown="1">

```
x= 4.0  
y= 6.0  
```

</div>

## Linear Programming: Solver(Gurobi, CPLEX, GLPK) 설치

1. Gurobi (유료)
2. CPLEX (유료)
3. GLPK (무료)

## Linear Programming: Pyomo
- 설치: `pip install pyomo`
- 최적화 Framework만 제공하며, Solver는 따로 설치 필요하다.(필자는 GLPK Solver 셋업함)

```python
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

model.x = pyo.Var(bounds=(0,10))
model.y = pyo.Var(bounds=(0,10))

x = model.x
y = model.y

model.C1 = pyo.Constraint(expr= -x+2*y<=8)
model.C2 = pyo.Constraint(expr= 2*x+y<=14)
model.C3 = pyo.Constraint(expr= 2*x-y<=10)

model.obj = pyo.Objective(expr= x+y, sense=maximize)

opt = SolverFactory('glpk')
opt.solve(model)

model.pprint()

x_value = pyo.value(x)
y_value = pyo.value(y)

print('x=', x_value)
print('y=', y_value)
```

출력

<div class="no-highlight" markdown="1">

```
2 Var Declarations
    x : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   4.0 :    10 : False : False :  Reals
    y : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   6.0 :    10 : False : False :  Reals

1 Objective Declarations
    obj : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : maximize : x + y
  
3 Constraint Declarations
    C1 : Size=1, Index=None, Active=True
        Key  : Lower : Body      : Upper : Active
        None :  -Inf : - x + 2*y :   8.0 :   True
    C2 : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :  -Inf : 2*x + y :  14.0 :   True
    C3 : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :  -Inf : 2*x - y :  10.0 :   True

6 Declarations: x y C1 C2 C3 obj
x= 4.0
y= 6.0
```

</div>

## Linear Programming: PuLP

- 설치
  - `pip install cython`
  - `pip install pulp`

- 코드

```python
import pulp as pl

model = pl.LpProblem('Ex', pl.LpMaximize)

x = pl.LpVariable('x', 0, 10)
y = pl.LpVariable('y', 0, 10)

# Constraints
model += -x+2*y<=8
model += 2*x+y<=14
model += 2*x-y<=10

# Objective
model += x+y

status = model.solve()

x_value = pl.value(x)
y_value = pl.value(y)

print('x=', x_value)
print('y=', y_value)
```

<div class='no-highlight' markdown='1'>

```
    x= 4.0
    y= 6.0
```

</div>

## Which solver should I choose?

| Framework | Linear Problems | Nonlinear Problems | Easy | Configure Solver |
|-----------|-----------------|--------------------|------|---------------|
| Pyomo     | O               | O                  | O    | O             |
| Ortools   | O               | X                  | O    | X             |
| PuLP      | O               | X                  | O    | O             |
| SCIP      | O               | O                  | O    | X             |
| SciPy     | O               | O                  | X    | △            |

| Solver | Linear Problems | Nonlinear | Free / Commercial |
|--------|-----------------|-----------|-------------------|
| Gurobi | O               | X         | Commercial        |
| Cplex  | O               | X         | Commercial        |
| CBC    | O               | X         | Free              |
| GLPK   | O               | X         | Free              |
| IPOPT  | X               | O         | Free              |
| SCIP   | O               | O         | Free              |
| Baron  | X               | O         | Commercial        |


## Exercise
- Show the optimal solution for the following problem:


```python
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

model.x = pyo.Var(bounds=(None, 3))
model.y = pyo.Var(bounds=(0, None))

x, y = model.x, model.y

model.C1 = pyo.Constraint(expr= x+y<=8)
model.C2 = pyo.Constraint(expr= 8*x+3*y>=-24)
model.C3 = pyo.Constraint(expr= -6*x+8*y<=48)
model.C4 = pyo.Constraint(expr= 3*x+5*y<=15)

model.obj = pyo.Objective(expr= -4*x-2*y, sense=minimize)
opt = SolverFactory('glpk')
opt.solve(model)

model.pprint()
```

<div class='no-highlight' markdown='1'>

```
    2 Var Declarations
        x : Size=1, Index=None
            Key  : Lower : Value : Upper : Fixed : Stale : Domain
            None :  None :   3.0 :     3 : False : False :  Reals
        y : Size=1, Index=None
            Key  : Lower : Value : Upper : Fixed : Stale : Domain
            None :     0 :   1.2 :  None : False : False :  Reals
    
    1 Objective Declarations
        obj : Size=1, Index=None, Active=True
            Key  : Active : Sense    : Expression
            None :   True : minimize : -4*x - 2*y
    
    4 Constraint Declarations
        C1 : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :  -Inf : x + y :   8.0 :   True
        C2 : Size=1, Index=None, Active=True
            Key  : Lower : Body      : Upper : Active
            None : -24.0 : 8*x + 3*y :  +Inf :   True
        C3 : Size=1, Index=None, Active=True
            Key  : Lower : Body       : Upper : Active
            None :  -Inf : -6*x + 8*y :  48.0 :   True
        C4 : Size=1, Index=None, Active=True
            Key  : Lower : Body      : Upper : Active
            None :  -Inf : 3*x + 5*y :  15.0 :   True
    
    7 Declarations: x y C1 C2 C3 C4 obj
```

</div>


```python
print('x=', pyo.value(x))
print('y=', pyo.value(y))
```

<div class='no-highlight' markdown='1'>

```
    x= 3.0
    y= 1.2
```

</div>
