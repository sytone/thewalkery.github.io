---
title: "Optimization with Python 06 - Particle Swarm"
excerpt: "Optimization with Python 06 - Particle Swarm"
date: 2022-03-04 00:00:00 +0900
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

# Particle Swarm Optimization
- 군집 기반 최적화는 수리적 최적화의 한 방법론으로써, 군집 기반 최적화에서는 여러 개의 optimizer가 서로 정보를 교환하며 동시에 최적화를 수행한다.

# Example

## 문제 

Objective  
- $\max x + xy$

Constrains  
- $-x + 2yx \le 8$  
- $2x + y \le 14$  
- $2x - y \le 10$  
- $0 \le x \le 10$  
- $0 \le y \le 10$  
- $x$ integer  

## 파이썬 코드

```python
!pip install pyswarm
```

<div class="no-highlight" markdown="1">

```
    Collecting pyswarm
      Downloading pyswarm-0.6.tar.gz (4.3 kB)
      Preparing metadata (setup.py): started
      Preparing metadata (setup.py): finished with status 'done'
    Requirement already satisfied: numpy in c:\users\thexl\onedrive\desktop\pythonworkspace\optimization-with-python\.venv\lib\site-packages (from pyswarm) (1.21.5)
    Using legacy 'setup.py install' for pyswarm, since package 'wheel' is not installed.
    Installing collected packages: pyswarm
      Running setup.py install for pyswarm: started
      Running setup.py install for pyswarm: finished with status 'done'
    Successfully installed pyswarm-0.6
```

</div>

```python
import numpy as np
from pyswarm import pso

def model_obj(x):
    pen = 0
    x[0] = np.round(x[0], 0)
    if not -x[0] + 2*x[1]*x[0] <= 8:
        pen = np.inf
    if not 2*x[0] + x[1] <= 14:
        pen = np.inf
    if not 2*x[0] - x[1] <= 10:
        pen = np.inf
    
    return -(x[0]+x[0]*x[1]) + pen

def cons(x):
    # constraint를 여기에 써도 되나, model_obj에서 penalty처리헀으므로 빈 리스트 리턴
    return []

lb = [0, 0]
ub = [10, 10]
x0 = [0, 0] # Initial point

xopt, fopt = pso(model_obj, lb, ub, x0, cons)
```

<div class="no-highlight" markdown="1">

```
    Stopping search: Swarm best objective change less than 1e-08
```

</div>

```python
print('x =', xopt[0])
print('y =', xopt[1])
```

<div class="no-highlight" markdown="1">

```
    x = 5.0
    y = 1.299999992890282
```

</div>

# Resources

- [입자 군집 최적화 (Particle Swarm Optimization, PSO)의 개념과 구현](https://untitledtblog.tistory.com/172)
- [Particle Swarm Optimization from Theory to Applications]({{site.baseurl}}/assets/resources/2022-03-04-optimization-with-python-06-PSO-concepts.pdf)