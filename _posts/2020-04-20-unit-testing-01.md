---
title: "Unit Testing and Test Driven Development in Python"
date: 2020-04-20 20:00:00 +0900
categories:
  - Unit Testing
---

## Unit Testing and Test Driven Development in Python

## What is unit testing?

### Why do we unit test?

- software bugs hurt the business!
- software testing catches the bugs before they get to the field
- Need several levels of safety nets

### Levels of testing

- **Unit Tesxting** - Testing at the function level.
- **Component Testing** - Testing is at the library and compiled binary level
- **System Testing** - Tests the external interfaces of a system which is a collection of sub-systems.
- **Performance Testing** - Testing done at sub-system and system levels to verify timing and resource usages are acceptable.

### Unittesting specifics

- Tests individual functions
- A test should be written for each test case for a fucntion(all positive and negative test cases).
- Groups of tests can be combined into test suites for better organiztion.
- Executes in the development environment rather than the production evironment.
- Execution of the tests should be automated.

### A Simple Example

```python
import pytest

# Production Code
def str_len(theStr):
    return len(theStr)

# A Unit Test
def test_string_length():
    testStr = "1" # Setup
    result = str_len(testStr) # Action
    assert result == 1 # Assert
```

### Summary

- Unit tests are the first safety net for catching bugs before they get to the field.
- Unit tests validate test cases for individual functions.
- They should build and run in the developer's development environment.
- Unit tests should run fast!

## Overview of Test-Driven Development

### What is test-driven development?

- A process where the developer takes personal responsibility for the quality of their code.
- Unit tests are written *before* the production code.
- Don't write all the tests or production code at once.
- Tests and production code are both written together in small bit of functionality.

### What are some of the benefits of TDD?

- Gives you the confiden e to change the code.
- Gives you immediate feedback
- Documents what the code is doing
- Drives good object oriented design

### TDD Beginnings

- Created by Kent Beck in the 1990's as part of the Extreme Programming software developemnt process.

### TDD Work flow: RED, GREEN, REFACTOR

- Write a failing unit test (the RED phase)
- Write just enough production code to make that test pass (the GREEN phase)
- Refactor the unit test and the production code to make it clean (the REFACTOR phase)
- Repeat until the feature is complete.

### Uncle Bob's 3 laws of TDD

- You may not write any production code until you have written a failing unit test.
- You may not write more of a unit test than is sufficient to fail, and not compiling is failing.
- You may not write more production code than is sufficient to pass the currently failing unit test.

### Example

#### Test1 -> Success

```python
import pytest

def test_canAssertTrue():
    assert True
```

#### Test2 -> Fail (Error)

```python
import pytest

def test_canCallFizzBuzz():
    fizzBuzz(1)
```

#### Test3 -> Success

```python
import pytest

def fizzBuzz(value):
    return

def test_canCallFizzBuzz():
    fizzBuzz(1)
```

#### Test4 -> Fail

```python
import pytest

def fizzBuzz(value):
    return

def test_canCallFizzBuzz():
    fizzBuzz(1)

def test_return1WithPassed():
    retVal = fizzBuzz(1)
    assert retVal == "1"
```

#### TestX -> Fail

```python
import pytest

def fizzBuzz(value):
    return "1"

def test_fizzBuzz():
    assert fizzBuzz(1) == "1"

def test_fizzBuzz2():
    assert fizzBuzz(2) == "2"
```

#### TestX -> Success

```python
import pytest

def fizzBuzz(value):
    return str(value)

def test_fizzBuzz():
    assert fizzBuzz(1) == "1"

def test_fizzBuzz2():
    assert fizzBuzz(2) == "2"
```

#### Refactor Phase

```python
import pytest

def fizzBuzz(value):
    return str(value)

def checkFizzBuzz(value, expected):
    retVal = fizzBuzz(value)
    assert retVal == expected

def test_fizzBuzzWith1():
    checkFizzBuzz(1, "1")

def test_fizzBuzzWith2():
    checkFizzBuzz(2, "2")
```

#### Red Phase

```python
import pytest

def fizzBuzz(value):
    return str(value)

def checkFizzBuzz(value, expected):
    retVal = fizzBuzz(value)
    assert retVal == expected

def test_fizzBuzzWith1():
    checkFizzBuzz(1, "1")

def test_fizzBuzzWith2():
    checkFizzBuzz(2, "2")

def test_fizzBuzzWith3():
    checkFizzBuzz(3, "Fizz")

def test_fizzBuzzWith5():
    checkFizzBuzz(5, "Buzz")
```

#### Green Phase

```python
import pytest

def fizzBuzz(value):
    if (value % 3) == 0:
        return "Fizz"
    if (value % 5) == 0"
        return "Buzz"
    return str(value)

def checkFizzBuzz(value, expected):
    retVal = fizzBuzz(value)
    assert retVal == expected

def test_fizzBuzzWith1():
    checkFizzBuzz(1, "1")

def test_fizzBuzzWith2():
    checkFizzBuzz(2, "2")

def test_fizzBuzzWith3():
    checkFizzBuzz(3, "Fizz")

def test_fizzBuzzWith5():
    checkFizzBuzz(5, "Buzz")
```

#### Refactor Phase

```python
import pytest

def fizzBuzz(value):
    if isMultiple(value, 3):
        return "Fizz"
    if isMultiple(value, 5):
        return "Buzz"
    return str(value)

def isMultiple( value, mod ):
    return (value % mod) == 0

def checkFizzBuzz(value, expected):
    retVal = fizzBuzz(value)
    assert retVal == expected

def test_fizzBuzzWith1():
    checkFizzBuzz(1, "1")

def test_fizzBuzzWith2():
    checkFizzBuzz(2, "2")

def test_fizzBuzzWith3():
    checkFizzBuzz(3, "Fizz")

def test_fizzBuzzWith5():
    checkFizzBuzz(5, "Buzz")
```

#### Red Phase

```python
import pytest

def fizzBuzz(value):
    if isMultiple(value, 3):
        return "Fizz"
    if isMultiple(value, 5):
        return "Buzz"
    return str(value)

def isMultiple( value, mod ):
    return (value % mod) == 0

def checkFizzBuzz(value, expected):
    retVal = fizzBuzz(value)
    assert retVal == expected

def test_fizzBuzzWith1():
    checkFizzBuzz(1, "1")

def test_fizzBuzzWith2():
    checkFizzBuzz(2, "2")

def test_fizzBuzzWith3():
    checkFizzBuzz(3, "Fizz")

def test_fizzBuzzWith5():
    checkFizzBuzz(5, "Buzz")

def test_fizzBuzzWith15():
    checkFizzBuzz(15, "FizzBuzz")
```
