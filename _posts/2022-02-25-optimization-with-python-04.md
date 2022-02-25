---
title: "Optimization with Python 04 - Pyomo"
excerpt: "Optimization with Python 04 - Pyomo"
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

# Working with Pyomo
  
## Using different solvers
- `pyomo help --solvers` 명령어를 통해 지원하는 Solver들을 볼 수 있다.

### CBC
- CBC Solver를 설치하자.

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

# PATH 환경 변수 추가 안하고 직접 exe 경로 입력 가능
opt = SolverFactory('cbc', executable='D:\\Solver-CBC\\win32-msvc9\\bin\\cbc.exe')

opt.solve(model)
```

<div class="no-highlight" markdown="1">

```
    {'Problem': [{'Name': 'unknown', 'Lower bound': -14.4, 'Upper bound': -14.4, 'Number of objectives': 1, 'Number of constraints': 5, 'Number of variables': 3, 'Number of nonzeros': 2, 'Sense': 'minimize'}], 'Solver': [{'Status': 'ok', 'User time': -1.0, 'System time': 0.0, 'Wallclock time': 0.0, 'Termination condition': 'optimal', 'Termination message': 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.', 'Statistics': {'Branch and bound': {'Number of bounded subproblems': None, 'Number of created subproblems': None}, 'Black box': {'Number of iterations': 1}}, 'Error rc': 0, 'Time': 0.026488542556762695}], 'Solution': [OrderedDict([('number of solutions', 0), ('number of solutions displayed', 0)])]}
```

</div>

```python
model.pprint()
```

<div class="no-highlight" markdown="1">

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

<div class="no-highlight" markdown="1">

```
    x= 3.0
    y= 1.2
```

</div>

## Arrays and Summations

예시를 통해 배우자.  
간단한 전기 시스템의 예를 들어 보자.  

- Power Generation (Pg)
  - 발전기

| ID | Cost | Power Generation |
|----|------|------------------|
| 0  | 0.10 |            20 kW |
| 1  | 0.05 |            10 kW |
| 2  | 0.30 |            40 kW |
| 3  | 0.40 |            50 kW |
| 4  | 0.01 |             5 kW |

- Load Points (Pd)
  - 부하

| ID | Load Demand |
|----|-------------|
| 0  |       50 kW |
| 1  |       20 kW |
| 2  |       30 kW |

- 제약 조건
  -  Only generators 0 and 3 can provide power to load point 0

이를 Modeling해보면,  
- Terms:
  - $C_g(i_g)$: $i_g$ 번째 발전기의 전기 생산 비용
  - $P_g(i_g)$: $i_g$ 번째 발전기의 전기 생산량
  - $P_d(i_d)$: $i_d$ 번째 부하의 전기 요구량

- Objective:
  - $min \sum_{i_g=4}^4 C_g(i_g)P_g(i_g)$
    - 목표(Minimize): ($i_g$ 번째 발전기 생산량 x 비용) x 모든 $i_g$에 대해서 Sum
- Constraints:
  - $\sum_{i_g=0}^4 P_g(i_g)=\sum_{i_c=0}^2 P_d(i_d)$
    - 제약: 모든 발전기의 생산량 합 = 모든 부하의 요구량 합
  - $P_g(i_g) \ge 0\qquad \forall i_g$
  - $P_g(i_g) \le P_g(i_g)^{LIM}\qquad \forall i_g$
    - 제약: 각 발전기의 발전량이 Limit 이하여야 함

## 실습

```python
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd

# inputs
dataGen = pd.read_excel('10_Pyomo_Summations_Data.xlsx', sheet_name='gen')
dataLoad = pd.read_excel('10_Pyomo_Summations_Data.xlsx', sheet_name='load')

dataGen
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>limit</th>
      <th>cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>40</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>50</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataLoad
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
Ng = len(dataGen)

model = pyo.ConcreteModel()

model.Pg = pyo.Var(range(Ng), bounds=(0,None)) # 5개 Var 생성(Ng=5)
#참고. 아래 Command는 5x3 Dimension의 Var 생성 가능하다.
#model.Pg = pyo.Var(range(Ng), range(3), bounds=(0,None))

model.Pg.pprint()
```

<div class="no-highlight" markdown="1">

```
    Pg : Size=5, Index=Pg_index
        Key : Lower : Value : Upper : Fixed : Stale : Domain
          0 :     0 :  None :  None : False :  True :  Reals
          1 :     0 :  None :  None : False :  True :  Reals
          2 :     0 :  None :  None : False :  True :  Reals
          3 :     0 :  None :  None : False :  True :  Reals
          4 :     0 :  None :  None : False :  True :  Reals
```

</div>


```python
Pg = model.Pg

[Pg[g] for g in dataGen['id']]
```


<div class="no-highlight" markdown="1">

```
    [<pyomo.core.base.var._GeneralVarData at 0x1a5e626b898>,
     <pyomo.core.base.var._GeneralVarData at 0x1a5e626b748>,
     <pyomo.core.base.var._GeneralVarData at 0x1a5e626b3c8>,
     <pyomo.core.base.var._GeneralVarData at 0x1a5e626b2e8>,
     <pyomo.core.base.var._GeneralVarData at 0x1a5e626b048>]
```

</div>

```python
print(sum([Pg[g] for g in dataGen['id']]))
```

<div class="no-highlight" markdown="1">

```
    Pg[0] + Pg[1] + Pg[2] + Pg[3] + Pg[4]
``` 

</div>

pg_sum = sum([Pg[g] for g in dataGen['id']])
# dataLoad['value']는 최적화 Variable이 아니므로 그냥 쓸 수 있음
model.balance = pyo.Constraint(expr = pg_sum == sum(dataLoad['value']))

model.cond = pyo.Constraint(expr = Pg[0] + Pg[3] >= dataLoad['value'][0])

model.limits = pyo.ConstraintList()
for g in dataGen['id']:
    model.limits.add(expr = Pg[g] <= dataGen.limit[g])

# objFcn
model.obj = pyo.Objective(expr = sum([Pg[g]*dataGen['cost'][g] for g in dataGen['id']]))
model.pprint()
```

<div class="no-highlight" markdown="1">

```
    2 Set Declarations
        Pg_index : Size=1, Index=None, Ordered=Insertion
            Key  : Dimen : Domain : Size : Members
            None :     1 :    Any :    5 : {0, 1, 2, 3, 4}
        limits_index : Size=1, Index=None, Ordered=Insertion
            Key  : Dimen : Domain : Size : Members
            None :     1 :    Any :    5 : {1, 2, 3, 4, 5}
    
    1 Var Declarations
        Pg : Size=5, Index=Pg_index
            Key : Lower : Value : Upper : Fixed : Stale : Domain
              0 :     0 :  None :  None : False :  True :  Reals
              1 :     0 :  None :  None : False :  True :  Reals
              2 :     0 :  None :  None : False :  True :  Reals
              3 :     0 :  None :  None : False :  True :  Reals
              4 :     0 :  None :  None : False :  True :  Reals
    
    1 Objective Declarations
        obj : Size=1, Index=None, Active=True
            Key  : Active : Sense    : Expression
            None :   True : minimize : 0.1*Pg[0] + 0.05*Pg[1] + 0.3*Pg[2] + 0.4*Pg[3] + 0.01*Pg[4]
    
    3 Constraint Declarations
        balance : Size=1, Index=None, Active=True
            Key  : Lower : Body                                  : Upper : Active
            None : 100.0 : Pg[0] + Pg[1] + Pg[2] + Pg[3] + Pg[4] : 100.0 :   True
        cond : Size=1, Index=None, Active=True
            Key  : Lower : Body          : Upper : Active
            None :  50.0 : Pg[0] + Pg[3] :  +Inf :   True
        limits : Size=5, Index=limits_index, Active=True
            Key : Lower : Body  : Upper : Active
              1 :  -Inf : Pg[0] :  20.0 :   True
              2 :  -Inf : Pg[1] :  10.0 :   True
              3 :  -Inf : Pg[2] :  40.0 :   True
              4 :  -Inf : Pg[3] :  50.0 :   True
              5 :  -Inf : Pg[4] :   5.0 :   True
    
    7 Declarations: Pg_index Pg balance cond limits_index limits obj
```

</div>


```python
opt = SolverFactory('cbc', executable='D:\\Solver-CBC\\win32-msvc9\\bin\\cbc.exe')
results = opt.solve(model)

dataGen['Pg'] = [pyo.value(Pg[g]) for g in dataGen['id']]

dataGen
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>limit</th>
      <th>cost</th>
      <th>Pg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20</td>
      <td>0.10</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10</td>
      <td>0.05</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>40</td>
      <td>0.30</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>50</td>
      <td>0.40</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0.01</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>

