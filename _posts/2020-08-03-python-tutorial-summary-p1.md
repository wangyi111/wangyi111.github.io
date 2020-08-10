---
title: 'A summary of python tutorial (P1)'
permalink: /posts/2020/08/python-tutorial-summary-p1/
categories: programming
tags:
  - python
  - tutorial
toc: true
toc_label: "CONTENT"
---

Learning a programming language needs continuous practice and also, frequent review. To make it convenient to look back and check the basics, I write this summary based on [Xuefeng Liao's python tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400).

**note**: Some parts may be too brief to understand without previous knowledge, please see [full tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400) for an addition.

# Part 1: Basic Grammar

## 01: data structure basics  
1. string bracket `""` or `''`
2. escape character `/`
3. Bool value: `True`, `False`  Bool operation: `and`, `or`, `not` 
4. Division: 10/3 = 3.333333  10//3 = 3  10%3 = 1
5. `#!/usr/bin/env python3`  `# -*- coding: utf-8 -*-`  (avoid gibberish when not using english)
6. format output: `print('%2d-%02d-%.2f' % (3, 1, 3.14159265))`  '%' = %%
7. list: classmates = ['A','B','C'] # changeable list  
    ```
    classmates[0] == 'A'  
    classmates[-1] == 'C'  
    classmates.append('D')  
    classmates.insert(1,'a')  
    classmates.pop() # delete last element   
    classmates.pop(1) # delete [1]   
    ```
8. tuple: classmates = ('A','B','C') # unchangeable tuple  
    ``` 
    classmates[0] == 'A'   
    t = (1,) # tuple with only one element   
    t = () # empty tuple   
    ```
9. conditional judgment:   
    ``` 
    if <>:   
        ...   
    elif <>:   
        ...   
    else:   
    ```
10. input: `s = input('input: ') # s is a string`
11. loop   
    for loop: `for x in range(5) # 0,1,2,3,4 `  
    while loop: `while <> `  
12. dict: d = {'A':1,'B':2,'C':3} # keys must be unchangeable: usually a string   
    ```
    d['A'] == 1   
    'B' in d # True   
    d.get('D',-1) # return -1   
    d.pop('C') # delete 'C':3   
    ```
13. set: s = set([1,2,3]) # no order   
    ```
    s = set([1,2,3,1,2]) # s==set([1,2,3]) same elements removed    
    s.add(4)   
    s.remove(4)   
    s1 & s2 # intersection   
    s1 | s2 # union set 
    ```

## 02: functions
1. empty function: `def fun(): pass   return x,y # return a tuple`  
2. function parameters    
    ```
    def fun(a,b,c=0): default parameter # default parameters in the end, and unchangeable type   
    def fun(*numbers): changeable parameter numbers # seen as a tuple, '*[1,2,3]' works   
    def fun(a,b,**kw): keyword(optional) parameter # seen as a dict, '**{'x':0,'y':1}' works   
      if 'x' in kw # check keyword parameter     
    def fun(a,b,*,x=0,y): naming KW parameter # fun(a,b,y=1)     
    def fun(a,b,*args,x,y) # x,y are KW parameter     
    ```
3. combine different types of parameters   
    #order: required->default->changeable->naming keyword->keyword   
    `def f1(a,b,c=0,*args,**kw) # args:tuple  kw:dict `    
4. recursive function: Hanoi tower problem   
```
    def Hanoi(n,a,b,c):   
        if n == 1:   
            print('move',a,'-->',c)   
        else:  
            Hanoi(n-1,a,c,b)  
            Hanoi(1,a,b,c)   
            Hanoi(n-1,b,a,c)   
```

## 03: advanced operations
1. slice: L = ['a','b','c','d']    
    ```
    L[0:3] == L[:3] # ['a','b','c'] first    
    L[-2:] # ['c','d'] last    
    L[-2:-1] # ['c']    
    L[0:3:2] # ['a','c'] step 2    
    L[:] # copy L    
    #the same for tuple and string    
    ```
2. iteration    
    ```
    isinstance('abc', Iterable) # see if sth is iterable, from collections import Iterable    
    for i,value in enumerate(['a','b','c']) # index-value pair    
    for x,y in [(1,1),(2,2),(3,3)]    
    ```
3. list comprehensions    
    ```
    list(range(5)) # [0,1,2,3,4]    
    [m+n for m in 'AB' for n in 'XY'] # ['AX','AY','BX','BY']    
    [d for d in os.listdir('.')] # list files and folders   
    for k,v in d.items() # d==['x':1,'y':2]   
    [s.lower() for s in L] # L==['He','Me']   
    [x for x in range(1,11) if x%2==0]   
    [x if x%2==0 else -x for x in range(1,11)]   
    ```
4. generator    
    g = (x*x for x in range(1,11)) # next() or for n in g   
```
    def fibonacci(max): # fibonacci series
        n,a,b = 0,0,1
        while n<max:
              yield b
              a,b = b,a+b
              n = n+1
        return 'done' # use StopIteration to get return
```
```
    def triangles(): # Pascal's triangle
        t=[1]
        while True:
            yield t
            t1 = t[:]
            t2 = t[:]
            t1.insert(0,0)
            t2.append(0)
            t = [t1[i]+t2[i] for i in range(len(t1))]
```
5. iterator # works with next()    
```
    isinstance((x for x in range(10)), Iterator) # True: generator is iterator   
    iter('abc') # transfer to iterator   
```

## 04: functional programming
1. high-order function (function as input parameter)  
    map(lambda x: x*x, [1,2,3]) # [1,4,9]    
    reduce(f,[x1,x2,x3,x4]) # f(f(f(x1,x2),x3),x4)    
```
    # example: string to number
    from functools import reduce
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    def char2num(s):
        return digits[s]
    def str2int(s):
        return reduce(lambda x,y: x*10+y, map(char2num,s))
```   
    list(filter(lambda x: x%2==1, [1,2,3,4,5])) # [1,3,5] filter returns an iterator      
```
    # example: generate all primes
    def _odd_iter():
        n = 1
        while True:
            n = n + 2
            yield n
    def _not_divisible(n):
        return lambda x: x%n>0
    def primes():
        yield 2
        it = _odd_iter()
        while True:
            n = next(it)
            yield n
            it = filter(_not_divisible(n),it)

    for n in primes(): # print all primes smaller than 1000
        if n < 1000:
            print(n)
        else:
            break
```  
```
    # example: filter all recursive numbers
    def is_palindrome(n):
        s = str(n)
        return s == s[::-1]
    output = filter(is_palindrome,range(1,1000))
    print(list(output))
``` 
    sorted(['a','c','B','z'], key=str.lower, reversed=True) # ['a','B','c','z']    
2. return function (closure)
```
    # example: each time returns a progressive increased number
    def createCounter():
        i = 0
        def counter():
            nonlocal i
            i = i + 1
            return i
        return counter()
```
3. anonymous function   
```
    # example: anonymous function
    def is_odd(n):
        return n%2==1
    L = list(filter(is_odd,range(1,20)))
    L = list(lambda x: x%2==1, range(1,20))
```
4. decorator
```
    import functools
    def log(text):
        def decorator(func):
            @functools.wraps(func) # wrapper.__name__ = func.__name__
            def wrapper(*args,**kw):
                print('%s %s():' % (text, func.__name__))
                return func(*args,**kw)
            return wrapper
        return decorator

    @log('excute') 
    def now():
        prit('2020-07-22')

    now() # log('excute')(now)
```  
```
    # example: print function excuting time
    import time,functools
    def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args,**kw):
        t1 = time.time()
        tmp = fn(*args,**kw)
        t2 = time.time()
        print('%s executed in %s s' % (fn.__name__, t2-t1))
        return tmp
    return wrapper
```
5. partial function # fix part of function paramters    
    int2 = functools.partial(int, base=2)  # int2(x) == int(x,base=2)

## 05: module
1. create a module
```
    #!/usr/bin/env python3
    #-*- coding: utf-8 -*-
    'a test module' # doc of this module: the first string of a module
    __author__ = 'Yi' # author

    import sys # python built-in module

    def test():
        args = sys.argv
        if len(args)==1:
            print('hello world')
        elif len(args)==2:
            print('hello %s' % args[1])
        else:
            print('too many arguments!')

    if __name__ == '__main__': # True when excuting the file in command line: usually for testing
        test()
```
2. private function or variable    
    __xxx__ # special variable   
    _xxx or __xxx # private variable   
    xxx # public variable      
3. module structure    
    --module: my_module (a folder with 2 py files)   `usage: import my_module `  
    ----__init__.py # my_module   
    ----abc.py # my_module.abc   

...

...

(To be continued.)
