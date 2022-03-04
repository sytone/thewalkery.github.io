---
title: "Optimization with Python 05 - Genetic Algorithm(Pymoo)"
excerpt: "Optimization with Python 05 - Genetic Algorithm(Pymoo)"
date: 2022-02-27 00:00:00 +0900
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

(Non-linear Algorithm은 생략함)

# Genetic Algorithm: Introduction
- Flexible한 알고리즘이다.
- 진화 과정과 유사하다.
- Genetic Algorithm은 Exact model이 아니다.

## Genetic Alogrithm Process
1. 각각의 Individual을 염색체로 생각하고, 각 염색체는 Gene(유전체)들로 이루어져있다.
2. Fitness Evaluation을 통해 각 염색체의 점수를 매긴다.
3. Mutation과 Crossover 기법을 통해 다음 Generation의 Population을 구성한다.
4. 알고리즘에 따라 어떻게 Mutation과 Crossover를 적용하는지 다르다.
5. 위 과정을 반복한다.(특정 세대 수까지 or 개선이 없을 때 까지 or 충분히 좋은 솔루션에 도달할 때까지)

# MINLP: Genetic Algorithm
- 구현할 문제는 다음과 같다.
  - Objective:
    - $max x + xy$
  - Constraints:
    - $-x + 2yx \le 8$
    - $2x + y \le 14$
    - $2x - y \le 10$
    - $0 \le x \le 10$
    - $0 \le y \le 10$
    - $x\ integer$


```python
!pip install geneticalgorithm
```

<div class="no-highlight" markdown="1">

```
    Collecting geneticalgorithm
      Downloading geneticalgorithm-1.0.2-py3-none-any.whl (16 kB)
    Collecting func-timeout
      Downloading func_timeout-4.3.5.tar.gz (44 kB)
         ---------------------------------------- 44.3/44.3 KB 2.3 MB/s eta 0:00:00
      Preparing metadata (setup.py): started
      Preparing metadata (setup.py): finished with status 'done'
    Requirement already satisfied: numpy in ...venv\lib\site-packages (from geneticalgorithm) (1.21.5)
    Using legacy 'setup.py install' for func-timeout, since package 'wheel' is not installed.
    Installing collected packages: func-timeout, geneticalgorithm
      Running setup.py install for func-timeout: started
      Running setup.py install for func-timeout: finished with status 'done'
    Successfully installed func-timeout-4.3.5 geneticalgorithm-1.0.2
```

</div>


```python
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def f(x):
    # Minimize 기준이므로 마이너스(-) 붙임
    pen = 0 # Penalty
    if not -x[0]+2*x[1]*x[0] <= 8:
        # 위 Constraint를 만족하지 못하면 penalty
        pen = np.inf
    if not 2*x[0]+x[1] <= 14:
        pen = np.inf
    if not 2*x[0]-x[1] <= 10:
        pen = np.inf
    return -(x[0]+x[0]*x[1]) + pen

varbounds = np.array([[0, 10], [0, 10]])
vartype = np.array([['int'], ['real']])
model = ga(function=f, dimension=2, variable_type_mixed=vartype, variable_boundaries=varbounds)
model.run()
```

<div class="no-highlight" markdown="1">

```
    ||________________________________________________ 4.2% GA is running...

    ...\.venv\lib\site-packages\geneticalgorithm\geneticalgorithm.py:353: RuntimeWarning: invalid value encountered in subtract
      normobj=maxnorm-normobj+1
    

     The best solution found:                                                                           
     [5.         1.29950421]
    
     Objective function:
     -11.49752103823522
```

</div>


# Genetic Algorithm: Routing Problem

- 아래의 라우팅 문제를 Genetic Algorithm을 사용하여 풀어보자.

![routing problem]({{site.baseurl}}/assets/images/2022-02-27-routing-problem.png)

- 노드 1에서 노드 7까지 가장 짧은 Path로 가려고 한다.
- 예상 솔루션은 1→2→4→7이다.
  
```python
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
from io import StringIO

paths_str = """node_from	node_to	distance
1	2	220
1	3	1500
2	4	650
2	5	900
4	7	500
5	7	400
3	6	500
6	7	400
"""

paths = pd.read_csv(StringIO(paths_str), sep="\t")
paths
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
      <th>node_from</th>
      <th>node_to</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>220</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
      <td>650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>5</td>
      <td>900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7</td>
      <td>500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>7</td>
      <td>400</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>6</td>
      <td>500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>7</td>
      <td>400</td>
    </tr>
  </tbody>
</table>
</div>



- Terms
    - $x_{ij}$: 노드 i에서 노드 j로 가는 길 선택 시 1, 아니면 0
    - $D_{ij}$: 노드 i에서 노드 j 사이의 거리
- Objective: $minimize(\sum x_{ij} D_{ij})$
- Constraints:
    1. $\sum_{j} x_{ji} = \sum_{k} x_{ik} \quad \forall i \notin \{1, 7\}$
    2. $\sum_{i} x_{1i} = 1$
    3. $\sum_{i} x_{i7} = 1$

- 변경: $x_{ij}$를 Variable로 대신에 실제 연결된 Path만 고려하도록 변경(Variable 수 7x7=49개에서 8개로 감소)
- Path 8개에 대해 각 Path의 사용 여부를 함수의 인자로 받기 → $x_{ij}$ 형태로 mapping 후 Constraints 적용


```python
node_count = 7
path_count = len(paths)

def create_distance_table(paths):
    distance_table = np.array([ [ -1 for j in range(node_count + 1)] for i in range(node_count + 1) ])
    for index, row in paths.iterrows():
        distance_table[row['node_from'], row['node_to']] = row['distance']
    return distance_table

distance_table = create_distance_table(paths)
distance_table
```

<div class="no-highlight" markdown="1">

```
    array([[  -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
           [  -1,   -1,  220, 1500,   -1,   -1,   -1,   -1],
           [  -1,   -1,   -1,   -1,  650,  900,   -1,   -1],
           [  -1,   -1,   -1,   -1,   -1,   -1,  500,   -1],
           [  -1,   -1,   -1,   -1,   -1,   -1,   -1,  500],
           [  -1,   -1,   -1,   -1,   -1,   -1,   -1,  400],
           [  -1,   -1,   -1,   -1,   -1,   -1,   -1,  400],
           [  -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1]])
```

</div>

```python
def p_to_x(p):
    # p[i]: i번째 path 사용 여부(0: 미사용, 1: 사용)
    # x[i][j]: i → j 사용 여부(0: 미사용, 1: 사용)
    x = np.array([ [ 0 for j in range(node_count + 1)] for i in range(node_count + 1) ])

    for i, pv in enumerate(p):
        if pv == 1:
            node_from = paths.loc[i, 'node_from']
            node_to = paths.loc[i, 'node_to']
            x[node_from, node_to] = 1
    return x
            
p_to_x([0,1,0,1,0,0,0,0])
```

<div class="no-highlight" markdown="1">

```
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
```

</div>

```python
p_to_x([0,1,0,1,0,0,0,0]) * distance_table
```

<div class="no-highlight" markdown="1">

```
    array([[   0,    0,    0,    0,    0,    0,    0,    0],
           [   0,    0,    0, 1500,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0,  900,    0,    0],
           [   0,    0,    0,    0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0,    0,    0,    0],
           [   0,    0,    0,    0,    0,    0,    0,    0]])
```

</div>

```python
def f(p):
    res = 0
    penalty = 0
    x = p_to_x(p)
    
    # penalty를 단순 np.inf로 주면 학습이 잘 안 됨
    # 원하는 조건에 가까울 수록 낮은 페널티 부과하도록 개선
    for i in range(1, 1 + node_count):
        if i == 1:
            if np.sum(x[1, :]) != 1:
                penalty += np.abs(np.sum(x[1, :]) - 1) * 1000000
        elif i == 7:
            if np.sum(x[:, 7]) != 1:
                penalty += np.abs(np.sum(x[:, 7]) - 1) * 1000000
        else:
            if np.sum(x[:, i]) != np.sum(x[i, :]):
                penalty += np.abs(np.sum(x[:, i]) - np.sum(x[i, :])) * 1000000
    
    total_distance = np.sum(x * distance_table)
    
    return total_distance + penalty
```


```python
# Test 1 (1 → 2 → 5 → 7)
f([1, 0, 0, 1, 0, 1, 0, 0])
```

<div class="no-highlight" markdown="1">

```
    1520
```

</div>



```python
# Test 2 (Not connected)
f([0, 1, 0, 0, 0, 0, 1, 0])
```

<div class="no-highlight" markdown="1">

```
    2002000
```

</div>

```python
# Test 3 (1 → 3 → 6 → 7)
f([0, 1, 0, 0, 0, 0, 1, 1])
```

<div class="no-highlight" markdown="1">

```
    2400
```

</div>

```python
varbounds = np.array([[0, 1] for p in range(path_count)])
vartype = np.array([['int'] for p in range(path_count)])

algorithm_param = {
    'max_num_iteration': 500,
    'population_size':100,
    'mutation_probability':0.30,
    'elit_ratio': 0.10,
    'crossover_probability': 0.50,
    'parents_portion': 0.30,
    'crossover_type':'uniform',
    'max_iteration_without_improv':100
}

model = ga(function=f, dimension=path_count, variable_type_mixed=vartype, variable_boundaries=varbounds, algorithm_parameters=algorithm_param)

model.run()
```

<div class="no-highlight" markdown="1">

```
     The best solution found:                                                                           
     [1. 0. 1. 0. 1. 0. 0. 0.]
    
     Objective function:
     1370.0
```

</div>


# Route Problem: 강사 솔루션

```python
import pandas as pd, numpy as np
from geneticalgorithm import geneticalgorithm as ga
from io import StringIO
 
#inputs
#nodes = pd.read_excel('route_inputs.xlsx', sheet_name='nodes')
#paths = pd.read_excel('route_inputs.xlsx', sheet_name='paths')

nodes_str = """node	description
1	origin
2	middle point
3	middle point
4	middle point
5	middle point
6	middle point
7	destination
"""

paths_str = """node_from	node_to	distance
1	2	220
1	3	1500
2	4	650
2	5	900
4	7	500
5	7	400
3	6	500
6	7	400
"""

nodes = pd.read_csv(StringIO(nodes_str), sep="\t")
paths = pd.read_csv(StringIO(paths_str), sep="\t")

nVars = len(paths)
 
#fitness function
def f(x):
    pen = 0
    
    #constraint sum(x) == 1 (origin)
    node_origin = int(nodes.node[nodes.description=='origin'])
    if sum([x[p] for p in paths.index[paths.node_from==node_origin]]) != 1:
        pen += 1000000 * np.abs(sum([x[p] for p in paths.index[paths.node_from==node_origin]]) - 1)
    
    #constraint sum(x) == 1 (destination)
    node_destination = int(nodes.node[nodes.description=='destination'])
    if sum([x[p] for p in paths.index[paths.node_to==node_destination]]) != 1:
        pen += 1000000 * np.abs(sum([x[p] for p in paths.index[paths.node_to==node_destination]]) - 1)
    
    #constraint sum(x, in) == sum(x, out)
    for node in nodes.node[nodes.description=='middle point']:
        sum_in = sum([x[p] for p in paths.index[paths.node_to==node]])
        sum_out = sum([x[p] for p in paths.index[paths.node_from==node]])
        if sum_in != sum_out:
            pen += 1000000 * np.abs(sum_in - sum_out)
 
    #objective function and return
    objFun = sum([x[p] * paths.distance[p] for p in paths.index])
    return objFun + pen
 
#bounds and var type
varbounds = np.array([[0,1]]*nVars)
vartype = np.array([['int']]*nVars)
 
#GA parameters
algorithm_param = {'max_num_iteration': 500,\
                   'population_size':100,\
                   'mutation_probability':0.30,\
                   'elit_ratio': 0.10,\
                   'crossover_probability': 0.50,\
                   'parents_portion': 0.30,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':100}
 
#Solve
model = ga(function=f,
           dimension=nVars,
           variable_type_mixed=vartype,
           variable_boundaries=varbounds,
           algorithm_parameters=algorithm_param)
model.run()
 
#print
x = model.best_variable
objFun = model.best_function
paths['activated'] = 0
for p in paths.index:
    paths.activated[p] = x[p]
 
print('\n\nAll Paths:')
print(paths)
 
print('\nSelected Paths:')
print(paths[paths.activated==1])
 
print('\nTotal path:', objFun)
```

- Important points
  1. The x variable represents all the variables inside an array.
  2. pen is the penalization when a possible x solution does not meet the constraints. It starts as 0 and increases if constraints are not met. Remember, the fitness function will evaluate each random solution from the GA.
  3. Each constraint is checked, and if this constraint is not met, we define the penalization pen.
  4. The penalization is not a single number. Is 1.000.000 times the distance that the evaluated solution is from the expected constraint.
  5. Finally, we calculate the objective function of the problem and return this objective function and the penalization.


# Multi-Objective Problems using NSGA-II 
## NSGA-II: Introduction
- NSGA-II(Non-dominated Sorting Genetic Algorithm II)는 Multi-objective 문제를 푸는데 잘 알려진 알고리즘이다.

## Introduction to multi-objective problems
- multi-objective problem에서 모델링하는 objective function들이 동일한 속성인지 아닌지 알아야 한다.
- 만약 objective function들이 동일한 속성이라면(예를 들어 "금액") 단순히 objective function들을 더해서 하나의 objective function으로 만들수도 있다.
- 예를 들어 투자액을 최소화하면서 유지보수 비용을 최소화하고 싶다면 "투자액 + 유지보수 비용"이라는 하나의 objective function을 사용 가능하며, 이는 multi-objective problem이 아니게 된다.
- 하지만, 예를 들어 당신은 차에 대한 투자 비용을 줄이면서 차의 승차감을 최대화하고 싶다고 하자. 이 경우 평가가 단순하지 않다. 왜냐하면 승차감과 투자 비용은 서로 다른 속성이기 때문이다. 이 경우 두 objective function을 갖게 된다.
  1. Objective function 1: 투자 금액 최소화
  2. Objective function 2: 승차감 최대화
- Multi-objective problem 관련된 특정한 알고리즘과 컨셉이 있다. 중요한 컨셉은 pareto front이다. 당신이 두 개의 objective function f1과 f2가 있다고 하자(두 function 모두 minimize가 목표). 그러면 우리는 모든 가능한 솔루션과 Pareto Front를 아래 그림과 같이 그릴 수 있다.

![pareto front]({{site.baseurl}}/assets/images/2022-02-27-pareto-front.png)

- 모든 가능한 솔루션은 "objective function space" 혹은 "solution space"라고 정의된다. 그리고 Pareto Front는 non-dimonated set solution이다. non-dominated solution은 다른 더 나은 솔루션이 없는 솔루션들을 말한다. 위 그림에서 파란 점의 솔루션들 사이에서는 어떤 솔루션이 다른 어떤 솔루션보다 더 낫다고 말할 수 없다. 우리는 단지 파란 솔루션들이 빨간색 솔루션들보다는 더 좋다고 말할 수 있을 뿐이다. 
- 최고의 솔루션을 찾기 위해서는 Pareto Front에서 선택 전략을 설정할 수 있다. 예를 들어 (0, 0) 좌표에서 가장 가까운 점, 혹은 Pareto Front이 가운데 점 등.

## How to solve Multi-Objective Problems

- 가장 유명한 알고리즘 중 하나는 Genetic Algorithm의 변형인 NSGA-II이다. NSGA-II를 이용하여 multi-objective problem을 풀기 위해서는 Pymoo 패키지에 관해 읽어보길 바란다([pymoo.org](https://pymoo.org/)).
- Pymoo는 사용하기 쉬운 패키지로, 우리가 앞에서 배운 GA 개념과 매우 유사하다. 웹사이트의 예제들이 잘 되어있으므로 참고하면 좋다. 
- 먼저, `pip install pymoo` 명령어로 pymoo를 설치하자.
- 예제 문제는 다음과 같다.
- Objectives
  1. $\min f_1(x) = 100 (x_1^2 + x_2^2)$
  2. $\min f_2(x) = (x_1 - 1)^2 + x_2^2$
- Constraints
  - $g_1(x) = 2(x_1 - 0.1)(x_1 - 0.9) / 0.18 \le 0$
  - $g_2(x) = -20(x_1-0.4)(x_1-0.6) / 4.8 \le 0$
  - $-2 \le x_1 \le 2$
  - $-2 \le x_2 \le 2$
  - $x \in \mathbb{R}$


```python
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
 
#definition of the problem
class MyProblem(ElementwiseProblem):
     
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))
 
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2
 
        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8
 
        out["F"] = [f1, f2]
        out["G"] = [g1, g2]
 
problem = MyProblem()
 
#parameters
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)
 
#termination criteria
termination = get_termination("n_gen", 40)
 
#solve problem
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
 
#get solutions
X = res.X
F = res.F
```

- **f1**과 **f2**는 두 개의 minimize 해야하는 objective function이다. max problem을 min problem으로 변환하기 위해서는 단순히 -1을 곱해주면 된다. $\max f1 = \min -f1$
- **g1**과 **g2**는 constraint들이다. 반드시 ≤ operator를 사용해서 정의해야 한다. ≥ Constraint를 -1을 곱해서 ≤ 형태의 Constraint로 변경할 수 있다. 예를 들어 $g1 \ge A$는 $-g1 \le -A$로 변환할 수 있다.
- Equality(=) 조건에 대해서는 두 개의 Constraint를 사용하거나(≤ 와 ≥), 이전 강의에서 사용한 방식처럼 Penalization을 적용할 수도 있다.
- `super().__init__` 시 파라미터 정의가 필요하다.
  - `n_var`: 문제의 변수 갯수
  - `n_obj`: 문제의 Objective functino 갯수
  - `xl`: 변수 X의 lower bound
  - `xu`: 변수 X의 upper bound
- 휴리스틱/진화 알고리즘을 사용하는 많은 패키지에서는 모든 variable들을 하나의 array로 변환해야한다.
- 예를 들어, Z1, Z2, Z3, Y라는 Variable이 있다고 가정하자. 만약 Pyomo를 사용했다면 Z와 Y라는 두 가지 변수를 사용했겠지만, Pymoo에서는 X1, X2, X3, X4로 사용하자.(X1=Z1, X2=Z2, X3=Z3, Y=Z4)

## Best Solution 정의하기

두 개의 목적 함수를 가지고 있다면, Pareto Front를 그래프로 나타낼 수 있다.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()
```

![pareto front]({{site.baseurl}}/assets/images/2022-02-27-optimization-with-python-05-pareto-front.png)

Best Solution을 정의하기 위해서는 **X**와 **F**를 기준으로 사용할 수 있다. 예를 들어 (0, 0) 원점에 가장 가까운 점 (F1, F2)를 찾을 수 있다.