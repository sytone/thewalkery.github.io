---
title: "Optimization with Python 02 - Mathematical Modeling"
excerpt: "Optimization with Python 02 - Mathematical Modeling"
date: 2022-02-21 23:00:00 +0900
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

# Introduction to Mathematical Modeling
## What is Mathematical Modeling?
- $\min \sum_i Cost(i)$
- $Cost(i) = ...$

## How Do We Solve Optimization Problems?
  1. 문제 이해
  2. 문제 모델링
  3. 문제 풀이(컴퓨터 이용, Framework / Solver)
  4. 결과

## Type of Variables
- $\min C_1 + C_2$
- $C_1 = 0.1P_1^2+0.5$
- $C_2 = 0.1P_2+3$
- $P_1+P_2=P_T$

- Variable은 우리가 모르는 변수, 주로 최적화 대상
- Parameter는 우리가 알고 있는 값(예를 들어 $P_T$가 정해져 있을 수 있다)

### Variable 타입
- 주로 3가지 타입 사용한다
  - Continuous
  - Integer (discrete)
  - Binary (0, 1)
  - Others

## Objective Function and Constraints
- Objective: 어떤 솔루션이 다른 솔루션보다 좋은지 결정하는 요소
- Constraints: 어떤 솔루션이 feasible한 지 정의하는 요소

![solution]({{site.baseurl}}/assets/images/2022-02-21-optimization-with-python-solution.png)

- Objecttive
  - $\max x+y$
- Constraints
  - $-x+2y\le8$
  - $2x+y\le14$
  - $2x-y\le10$
  - $x,y\ge0$

파란 영역만 Constraints를 만족하는 feasible set이다. 

## How to Model Your Problem?
- 예시를 통해 이해하자

### Example 1. 
Mayke는 100,000 USD를 가지고 있다. 

A) Low risk fund (연간 5% 수익률)  
B) Medium risk fund (연간 10% 수익률)  
C) High risk fund (연간 12% 수익률)  

Mayke는 자신의 투자금의 최대 10%를 High risk에,
자신의 투자금의 최대 20%를 Medium risk에 투자하려고 한다.

수익을 최대화하기 위해서 A, B, C에 투자금을 어떻게 분배할 수 있을까?

1. 내 답
- Objective
  - $\max A * 1.05 + B * 1.1 + C * 1.12$
- Constraints
  - $C \le 10000$
  - $B \le 20000$ 
  - $A+B+C=100000$
  - $A,B,C\ge0$

2. 풀이
- Variables and indexes
  - $R_A, R_B, R_C$: A,B,C 펀드에 투자했을 때 이익
  - $C_A, C_B, C_C$: A,B,C 펀드의 투자금
- Constrains
  - $C_A+C_B+C_C=100,000$
  - $R_A=0.05C_A$
  - $R_B=0.10C_B$
  - $R_C=0.12C_C$
  - $0 \le C_B \le 0.2*100,000$
  - $0 \le C_C \le 0.1*100,000$
- Objective Function
  - $\max(R_A+R_B+R_C)$

### Example 2.
Example 1에서 하나의 투자 fund 추가한 Version
Mayke는 100,000 USD를 가지고 있다.
A) Low risk fund (연간 5% 수익률)
B) Medium risk fund (연간 10% 수익률)
C) High risk fund (연간 12% 수익률)
D) Especial fund ($10^{-6}*C_D^2$)

Mayke는 자신의 투자금의 최대 10%를 High risk에,
자신의 투자금의 최대 20%를 Medium risk에 투자하려고 한다.
그리고 최대 30%를 Especial fund에 투자하려고 한다.

- Variables and indexes
  - $R_A, R_B, R_C, R_D$: A,B,C,D 펀드에 투자했을 때 이익
  - $C_A, C_B, C_C, C_D$: A,B,C,D 펀드의 투자금
- Constrains
  - $C_A+C_B+C_C+C_D=100,000$
  - $R_A=0.05C_A$
  - $R_B=0.10C_B$
  - $R_C=0.12C_C$
  - $R_D=10^{-6}*C_D^2$
  - $0 \le C_B \le 0.2*100,000$
  - $0 \le C_C \le 0.1*100,000$
  - $0 \le C_D \le 0.3*100,000$
- Objective Function
  - $\max(R_A+R_B+R_C+R_D)$
  - 또는 $\max \sum_{f \in F} R_f$, Set $F=\{A,B,C,D\}$
    
### Example 3.
당신의 3개의 큰 기계가 있는 신발 회사의 사장이다. 당신은 생산의 총 비용을 최소화하고 싶다.

각 기계의 생산 비용은 다음과 같다.  
A) $C_A=0.1P_A^2+0.5P_A+0.1$  
B) $C_B=0.3P_B+0.5$  
C) $C_C=0.01P_C^3$  
$C$는 $P$개의 제품을 생산하는데 드는 총 비용을 나타낸다.  
다음 달에 10,000켤레의 신발 수요가 있다고 하자. 총 비용을 줄이기 위해 각각의 기계에서 몇 개씩 신발을 생산하는 것이 좋을까?

- Variables
  - $C_A, C_B, C_C$ → A, B, C의 샌상 비용
  - $P_A, P_B, P_C$ → 각 기계의 생산량(정수)

- $F=\{A, B, C\}$
- Constraints
  - $0 \le P_f\ for\ \forall f \in F$
  - $\sum_{f \in F} P_f=10,000$
  - $C_A=0.1P_A^2+0.5P_A+0.1$
  - $C_B=0.3P_B+0.5$  
  - $C_C=0.01P_C^3$ 
- Objective
  - $minimize \sum_{f \in F} C_f$

그런데, A, B의 Cost를 보면 0.1, 0.5라는 상수항이 있다. 만약에 A나 B 제품을 하나도 생산하지 않는다면 해당 상수항의 Cost가 들지 않는다고 하자. 이 상황을 Model에 포함시켜보자.

- Variables
  - $C_A, C_B, C_C$ → A, B, C의 샌상 비용
  - $P_A, P_B, P_C$ → 각 기계의 생산량(정수)

- $F=\{A, B, C\}$
- Constraints
  - $0 \le P_f\ for\ \forall f \in F$
  - $\sum_{f \in F} P_f=10,000$
  - $C_A=0.1P_A^2+0.5P_A+\beta_A 0.1$
  - $C_B=0.3P_B+\beta_B 0.5$  
  - $C_C=0.01P_C^3$ 
  - $P_A \le \beta_A M$
  - $P_B \le \beta_B M$
- Objective
  - $minimize \sum_{f \in F} C_f$

$\beta_A$와 $\beta_B$는 0 또는 1의 값을 가질 수 있다. $\beta_A=0$ 으로 하여 0.1이라는 상수항을 0으로 만든다면, $P_A \le 0$을 만족해야 하므로 0이 되어야 한다. $\beta_A=1$이면 $P_A \le M$로 갯수의 제약이 없다.(여기서 M은 매우 큰 수/무한대로 생각하면 된다).

### Example 4.
A에서 B로 가는 길을 최소화하는 generic formulation을 만들라.

![route]({{site.baseurl}}/assets/images/2022-02-21-optimization-with-python-route.png)

숫자는 한 점에서 다른 점까지의 거리를 나타낸다.

- Variables
  - $x_{i, j}$ → Binary decision on connction point i to j
  - 위 그래프에는 총 7개의 $x_{i, j}$가 존재한다.
- Parameters
  - $D_{i, j}$ → i점에서 j점까지의 거리
- Sets
  - $\Omega_i^{in}$ → Set of nodes that connect to arcs entering node i
  - $\Omega_i^{out}$ → Set of nodes that connect to arcs exiting node i
  - Example: 
    - $\Omega_{P1}^{in}=\{A, P2\}$
    - $\Omega_{B}^{in}=\{P1, P3\}$
    - $\Omega_{P1}^{out}=\{P2, B\}$
    - $\Omega_{A}^{out}=\{P1, P2\}$
- Objective
  - $minimize \sum_{i, j} x_{i, j} D_{i, j}$
- Constrains
  - $\sum_{j \in \Omega_A^{out}} x_{A, j}=1$ : A에서 나가는 decision이 1개
  - $\sum_{i \in \Omega_B^{in}} x_{i, B}=1$ : B로 들어오는 decision이 1개
  - $\sum_{j \in \Omega_i^{out}} x_{i, j}=\sum_{j \in \Omega_i^{in}} x_{j, i}\qquad\forall i \notin \{A, B\}$  
  A나 B를 제외한 모든 노드 i에 대해서 i에서 나가는 decision 수와 i로 들어오는 decision 수가 동일해야 함

### Example 5.
Petter는 건설 회사를 가지고 있다. 그는 회사의 5개 팀을 아래 건설 업무에 할당해야 한다.

A. 수익 500. 1팀 필요  
B. 수익 4,000. 3팀 필요  
C. 수익 3,000. 2팀 필요  
D. 수익 2,000. 1팀 필요  
E. 수익 2,000. 5팀 필요  

수익을 최대화하기 위해 건설 업무를 선택하라.
- 각 건설은 1회만 수행 가능하다.
- 모든 건설이 선택되진 않을 것이다.

$X \in \{A, B, C, D, E\}$
- Variables
  - $\beta_x$: x 업무 수행 여부(0 또는 1)
- Parameters
  - $R_x$: x 업무의 수익
  - $T_x$: x 업무의 필요 팀 수
  - $T_{max}$: 총 팀 수(5)
- Objective
  - $maximize \sum_{x \in X} \beta_x R_x$
- Constrains
  - $\sum_{x \in X} \beta_x T_x \le T_{max}$

### Example 6.
앞의 문제를 확장: Constrains 추가
Petter는 건설 회사를 가지고 있다. 그는 회사의 5개 팀을 아래 건설 업무에 할당해야 한다.

A. 수익 500. 1팀 필요  
B. 수익 4,000. 3팀 필요  
C. 수익 3,000. 2팀 필요  
D. 수익 2,000. 1팀 필요  
E. 수익 2,000. 5팀 필요  

수익을 최대화하기 위해 건설 업무를 선택하라.
- 각 건설은 1회만 수행 가능하다.
- 모든 건설이 선택되진 않을 것이다.
- C는 A가 선택되었을 때만 선택 가능하다.
- D는 A와 C가 선택되었을 때만 선택 가능하다.

$X \in \{A, B, C, D, E\}$
- Variables
  - $\beta_x$: x 업무 수행 여부(0 또는 1)
- Parameters
  - $R_x$: x 업무의 수익
  - $T_x$: x 업무의 필요 팀 수
  - $T_{max}$: 총 팀 수(5)
- Objective
  - $maximize \sum_{x \in X} \beta_x R_x$
- Constrains
  - $\sum_{x \in X} \beta_x T_x \le T_{max}$
  - $\beta_C \le \beta_A$
  - $\beta_D \le \beta_A$
  - $\beta_D \le \beta_C$

마지막 두 Constraint 대신에 $\beta_D \le \beta_A * \beta_C$를 사용해도 동일하나, 두 변수가 곱해진 Non-linear 문제가 되어 풀기가 어려워진다.

### Example 7.
Mark는 3일 동안 참석할 고객과의 스케줄을 계획 중이다.

아래는 고객별 시간과 수익

A. 2시간, 200 USD 수익  
B. 3시간, 500 USD 수익  
C. 5시간, 300 USD 수익  
D. 2시간, 100 USD 수익  
E. 6시간, 1,000 USD 수익  
E. 4시간, 300 USD 수익  

Mark는 하루 6시간씩 3일씩 일해서 수익을 최대하고 하려고 한다.
일별 스케쥴을 짜보자.
- 이동 시간은 무시하라
- 하나의 스케쥴에는 한 번만 참석 가능하다

- Variables
  - $x_{j, d}$ - Binary decision on attending job j in day d
- Parameters
  - $P_j$ - job j의 수익
  - $D_j$ - job j의 시간
  - $Th$ - 하루 업무 시간(6)
- Constrains
  - $\sum_d x_{j, d} \le 1 \qquad\forall j$
  - $\sum_j D_j x_{j, d} \le Th \qquad\forall d$
- Objective
  - $maximize \sum_j \sum_d P_j x_{j, d}$


### Example 8.
Example 7의 Constrains 추가
Mark는 3일 동안 참석할 고객과의 스케줄을 계획 중이다.

아래는 고객별 시간과 수익

A. 2시간, 200 USD 수익  
B. 3시간, 500 USD 수익  
C. 5시간, 300 USD 수익  
D. 2시간, 100 USD 수익  
E. 6시간, 1,000 USD 수익  
E. 4시간, 300 USD 수익  

Mark는 하루 6시간씩 3일씩 일해서 수익을 최대하고 하려고 한다.
일별 스케쥴을 짜보자.
- 이동 시간은 무시하라
- 하나의 스케쥴에는 한 번만 참석 가능하다
- Mark는 하루에 최대 1개의 Job만 수행하고 싶다

- Variables
  - $x_{j, d}$ - Binary decision on attending job j in day d
- Parameters
  - $P_j$ - job j의 수익
  - $D_j$ - job j의 시간
  - $Th$ - 하루 업무 시간(6)
- Constrains
  - $\sum_d x_{j, d} \le 1 \qquad\forall j$
  - $\sum_j x_{j, d} \le 1 \qquad\forall d$
  - $\sum_j D_j x_{j, d} \le Th \qquad\forall d$
- Objective
  - $maximize \sum_j \sum_d P_j x_{j, d}$

## How to Learn More?
- Try to solve exercises from books(『Operations Research』)