---
title: 'A summary of python tutorial (P2)'
permalink: /posts/2020/08/python-tutorial-summary-p2/
categories: programming
tags:
  - python
  - tutorial
toc: true
toc_label: "CONTENT"
---

Learning a programming language needs continuous practice and also, frequent review. To make it convenient to look back and check the basics, I write this summary based on [Xuefeng Liao's python tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400).

**note**: Some parts may be too brief to understand without previous knowledge, please see [full tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400) for an addition.

# Part 2: Object Oriented Programming

## 06: object oriented programming
1. class and instance
```
    class Student(object): # object: base class (here no base class)
        def __init__(self, name, score):
            self.name = name
            self.score = score
        def print_score(self):
            print('%s: %s' % (self.name, self.score))
    Jack = Student('Jack',100) # define a Student object "Jack"
    Jack.print_score()
    Jack.age = 20 # add property
```
2. access limit    
    `self.__private_var` # private varaible    
    `def get_private_var(self): return self.__private_var` # access the var with function    
    `def set_private_var(self,private_var): self.__private_var = private_var` # change the var     
3. inheritance and polymorphism   
```
    # example
    class Animal(object):   #编写Animal类
        def run(self):
            print("Animal is running...")

    class Dog(Animal):  #Dog类继承Amimal类，没有run方法
        pass

    class Cat(Animal):  #Cat类继承Animal类，有自己的run方法
        def run(self):
            print('Cat is running...')

    class Car(object):  #Car类不继承，有自己的run方法
        def run(self):
            print('Car is running...')

    class Stone(object):  #Stone类不继承，也没有run方法
        pass

    def run_once(animal): # any object having run() function can access
        animal.run()

    run_once(Animal())
    run_once(Dog())
    run_once(Cat())
    run_once(Car())
    run_once(Stone())
```
4. object information   
    type(123) # int  
    type(abs) # builtin_function_or_method  
    type(fn) # types.FunctionType  
    isinstance(dd,Animal) # True: dd=Animal()  
    isinstance(123,(int,str)) # True  
    dir('ABS') # return all properties of 'ABS'  
    'ABS'.__len__() # len('ABS')  
    hasattr('ABS','__len__') # True  
    getattr('ABS','lower') # True  
    setattr('ABS','xx') # set a new property to object 'ABS'  
```
    # example
    def readImage(fp):
        if hasattr(fp, 'read'):
            return readData(fp)
        return None
```
5. instance property and class property   
    class Student(object): name = 'Student' # class property: Student.name == 'Student'   
    Jack = Student()  Jack.name = 'Jack' # instance property: Student.name == 'Student'   

## 07: advanced Object Oriented Programming
1. `__slots__` (limit possible properties)
```
    # example
    class student(object):
        __slots__ = ('name','age') # limit possible properties
    s = student()
    s.name = 'Jack'
    def set_age(self,age):
        self.age = age
    from types import MethodType
    s.set_age = MethodType(set_age, s) # add set_age() to object s only
    student.set_age = set_age # add set_age() to class student
```
2. @property (既能检查参数，又可以用类似属性这样简单的方式来访问类的变量)
```
    # example
    class Screen(object):
        @property
        def width(self):
            return self._width
        @width.setter
        def width(self,width):
            self._width = width
        @property
        def height(self):
            return self._height
        @height.setter
            def height(self,height):
            self._height = height
        @property
            def resolution(self):
            return self._width * self._height
    s = Screen()
    s.width = 1024
    s.height = 768
    print('resolution =', s.resolution)
```
3. MixIn (多重继承)  
    `class dog(Mammal, RunnableMixIn, PetMixIn)` # inherit multiple base classes   
4. customize a class  #ddd
    `__str__`: description when directly use the class
```
    # example
    class Student(object):
        def __init__(self, name):
            self.name = name
        def __str__(self): 
            return 'Student object (name=%s)' % self.name
        __repr__ = __str__ # description when use a class object
    s = Student('Michael') # 'Student object (name: Michael)'
```
    `__iter__`: make the class a generator  
```
    # example
    class Fib(object):
        def __init__(self):
            self.a, self.b = 0, 1 # 初始化两个计数器a，b
        def __iter__(self):
            return self # 实例本身就是迭代对象，故返回自己
        def __next__(self):
            self.a, self.b = self.b, self.a + self.b # 计算下一个值
            if self.a > 100000: # 退出循环的条件
                raise StopIteration()
            return self.a # 返回下一个值
    # usage
    for n in Fib():
        print(n)
```
    `__getitem__`: make the class simliar to list
```
    # example
    class Fib(object):
        def __getitem__(self, n):
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a
    # usage
    f = Fib()
    f[10] # 89
```
    others: `__setitem__`, `__detitem__`, ...   
    `__getattr__`: if properties not found in the class object
```
    # example
    class Chain(object):
        def __init__(self, path=''):
            self._path = path   
        def users(self, username=''):
            return Chain('%s/users/%s' % (self._path, username.lower()))
        def __getattr__(self, path):
            return Chain('%s/%s' % (self._path, path))
        def __str__(self):
            return self._path
        __repr__ = __str__
    # usage
    Chain().users('Machael').repos
```
    `__call__(self)`: directly use the class object. use callable() to check if callable     
5. enumerate class   
    Enum class
```
    # example 1
    from enum import Enum
    Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')) # 1-12
    for name, member in Month.__members__.items():
        print(name, '=>', member, ',', member.value) # start from 1
```
```
    # example 2
    from enum import Enum, unique
    @unique
    class Weekday(Enum):
        Sun = 0 # Sun的value被设定为0
        Mon = 1
        Tue = 2
        Wed = 3
        Thu = 4
        Fri = 5
        Sat = 6
    print(Weekday.Tue) # Weekday.Tue
    print(Weekday['Tue']) # Weekday.Tue
    print(Weekday.Tue.value) # 2
    print(Weekday(1)) # Weekday.Mon
```
6. metaclass  
    type(): creat a class
```
    def fn(self,name='world'):
        print('Hello %s' % name)
    Hello = type('Hello',(object,),dict(hello=fn))
    h = Hello() # h is a class Hello object
```
    metaclass: quite complicated, come back later...  
```
    # metaclass是类的模板，所以必须从`type`类型派生：
    class ListMetaclass(type):
        def __new__(cls, name, bases, attrs): # 当前准备创建的类的对象,类名,类继承的父类集合,类的方法集合
            attrs['add'] = lambda self, value: self.append(value)
            return type.__new__(cls, name, bases, attrs)

    class MyList(list, metaclass=ListMetaclass):
        pass
```

## 08: error, debug and test
1. error
    ```
    try:
        print('try...')
        r = 10 / int('2')
        print('result:', r)
    except ValueError as e: # all kinds of error are classes inherited from BaseException
        print('ValueError:', e)
    except ZeroDivisionError as e:
        print('ZeroDivisionError:', e)
    else:
        print('no error!')
    finally:
        print('finally...')
    print('END')
    ```
    error stack: traceback  
    logging: record error, continue running  
    ```
    import logging
    try:
        print(10/int('0'))
    except Exception as e:
        logging.exception(e)
    print('END') # this line still runs
    ```
    raise error 
    ```
    class costomError(ValueError):
        pass
    def foo(s):
        n = int(s)
        if n==0:
            raise costomError('invalid value: %s' %s)
        return 10/n
    try:
        foo('0')
    except ValueError as e:
        print('ValueError!')
        raise # if not will not trace the error
    ```
2. debug  
    method 1: `print()`  
    method 2: `assert <expression>, 'error message'`  to close assert: `python -O xx.py`
    method 3: `logging`  
```
    import logging
    logging.basicConfig(level = logging.INFO) # 4 levels: debug, info, warning, error
    n = int('0')
    logging.info('n = %d' %n)
    print(10/n)
```
    method 4.1: pdb `python -m pdb xxx.py # 1--view code, n--next line, p x--view var, q--quit`    
    method 4.2: set breakpoint `pdb.set_trace() # c--continue, p x--view var`    
    method 5: use IDE (pycharm, vscode, etc.)  
3. unit test  
```
    # example
    # mydict.py: costom class to be tested
    class Dict(dict): 
        def __init__(self, **kw):
            super().__init__(**kw)
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(r"'Dict' object has no attribute '%s'" % key)
        def __setattr__(self, key, value):
            self[key] = value

    # mydict_test.py: test the class
    import unittest
    from mydict import Dict
    class TestDict(unittest.TestCase):
        def test_init(self):
            d = Dict(a=1, b='test')
            self.assertEqual(d.a, 1)
            self.assertEqual(d.b, 'test')
            self.assertTrue(isinstance(d, dict))
        def test_key(self):
            d = Dict()
            d['key'] = 'value'
            self.assertEqual(d.key, 'value')
        def test_attr(self):
            d = Dict()
            d.key = 'value'
            self.assertTrue('key' in d)
            self.assertEqual(d['key'], 'value')
        def test_keyerror(self):
            d = Dict()
            with self.assertRaises(KeyError):
                value = d['empty']
        def test_attrerror(self):
            d = Dict()
            with self.assertRaises(AttributeError):
                value = d.empty

    # run test: `$ python -m unittest mydict_test` 
```
    `setUp(self)` & `tearDown(self)`: run before and after each test  
4. doctest  
    test in the annotation of the code file  
  ```
    # example myabs.py
    def abs(n):
        '''
        Function to get absolute value of number.
        Example:
        >>> abs(1)
        1
        >>> abs(-1)
        1
        >>> abs(0)
        0
        >>> abs('A')
        Traceback (most recent call last):
            ...
        ValueError
        '''
        if not isinstance(n,(int,float)):
            raise ValueError
        return n if n>=0 else -n
    if __name__ == '__main__':
        import doctest
        doctest.testmod()
  ```
    test: `$ python myabs.py` (if correct there will be no output)

## 09: IO (input/output) programming
1. file read and write  
```
    # example 1
    try:
        f = open('test.txt','r')
        s = f.read() # read all as a string
        print(s)
    finally:
        if f:
            f.close()
    # example 2
    with open('test.txt','wa') as f: # auto run open() and close()
        f.write('hello world')
```
    read(): read all, return a string  
    read(size): read part   
    readline(): read a line   
    readlines(): read all, return a list   
    'rb': read binarary file   
    open('test.txt','r',encoding='gbk',errors='ignore'): read not utf-8 file, ignore errors  
2. StringIO and BytesIO # operate in memory   
```
    from io import StringIO
    f = StringIO('hello\nhi\nbye\n') # create StringIO in memery
    while True:
        s = f.readline()
        if s == '':
            break
        print(s.strip())
    
    from io import BytesIO
    fb = BytesIO()
    fb.write('中文'.encode('utf-8'))
    # fb = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
    # fb.read()
    print(fb.getvalue())
```
3. operate file and directory  
    ```
    import os
    os.name # system type
    os.uname() # unavalibale on windows
    os.environ # environment variable
    os.environ.get('PATH')
    os.environ.get('x','default')
    os.path.abspath('.') # absolute path of current directory
    os.path.join('/Users/bin','testdir') # /Users/bin/testdir
    os.mkdir('/Users/bin/testdir') # create a dir
    os.rmdir('/Users/bin/testdir') # remove a dir
    os.path.split('/Users/bin/testdir/file.txt') # ('/Users/bin/testdir','file.txt')
    os.path.splitext('testdir/file.txt') #('testdir/file','.txt')
    os.rename('file.txt','new.txt')
    os.remove('new.txt')
    import shutil # an addition to os module
    [x for x in os.listdir('.') if os.path.isdir(x)] # list all paths in current path
    [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.py'] # list .py  

    # example: find all files whose name includes a keyword
    def find_filename(folder,keyword):
        file_list = []
        for all_folder, all_folder_name, all_filename in os.walk(folder):
            for filename in all_filename:
                if keyword.lower() in filename.lower():
                    file_list.append(os.path.join(all_folder,filename))
        return file_list
    file_list = find_filename('.','test')
    print(file_list)   
    ```
4. pickling (serialization): store or transfer variables from memory  
    method 1: `import pickle`    `d = dict(name='Jack',age=20)`    `pickle.dumps(d)`   
    method 2: JSON -- a format (string) suits all languages
    ```
    import json
    class Student(object):
        def __init__(self,name,age):
            self.name = name
            self.age = age
    s = Student('Jack',20)
    js = json.dumps(s,default=lambda obj:obj.__dict__) # transfer to json format
    def dict2student(d):
        return Student(d['name'],d['age'])
    sj = json.loads(js,object_hook=dict2student) # transfer back to class Student object
    ```

## 10: process and thread
1. multiprocessing  
    `fork()` returns subprocess iD for parentprocess, 0 for subprocess;  unix only
    ```
    import os
    print('process (%s) start...' % os.getpid())
    # only works on unix/linux/mac
    pid = os.fork()
    if pid == 0:
        print(' this is child process (%s) whose parent is (%s).' % (os.getpid(),os.getpid()))
    else:
        print('child process (%s) created from (%s).' % (os.getpid(),pid))
    ```
    `multiprocessing` module for all platforms
    ```
    from multiprocessing import Process
    import os
    # 子进程要执行的代码
    def run_proc(name):
        print('Run child process %s (%s)...' % (name, os.getpid()))
    if __name__=='__main__':
        print('Parent process %s.' % os.getpid()) # parent process id
        p = Process(target=run_proc, args=('test',)) # create subprocess
        print('Child process will start.')
        p.start() # start subprocess
        p.join() # wait untill subprocess is finished
        print('Child process end.')
    ```
    `pool`: start multiple subprocesses
    ```
    from multiprocessing import Pool
    import os, time, random

    def long_time_task(name):
        print('Run task %s (%s)...' % (name, os.getpid()))
        start = time.time()
        time.sleep(random.random() * 3)
        end = time.time()
        print('Task %s runs %0.2f seconds.' % (name, (end - start)))

    if __name__=='__main__':
        print('Parent process %s.' % os.getpid())
        p = Pool(4) # create subprocess pool
        for i in range(5):
            p.apply_async(long_time_task, args=(i,)) # create subprocess
        print('Waiting for all subprocesses done...')
        p.close() # close pool, cannot add new subprocess
        p.join() # wait untill all subprocesses done
        print('All subprocesses done.')
    ```
    `subprocess`: control I/O of subprocess
    ```
    import subprocess
    print('$ nslookup')
    p = subprocess.Popen(['nslookup'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate(b'set q=mx\npython.org\nexit\n')
    print(output.decode('utf-8'))
    print('Exit code:', p.returncode)
    ```
    communication between processes: multiprocessing.Queue/Pipes
    ```
    from multiprocessing import Process, Queue
    import os, time, random
    # 写数据进程执行的代码:
    def write(q):
        print('Process to write: %s' % os.getpid())
        for value in ['A', 'B', 'C']:
            print('Put %s to queue...' % value)
            q.put(value)
            time.sleep(random.random())
    # 读数据进程执行的代码:
    def read(q):
        print('Process to read: %s' % os.getpid())
        while True:
            value = q.get(True)
            print('Get %s from queue.' % value)
    if __name__=='__main__':
        # 父进程创建Queue，并传给各个子进程：
        q = Queue()
        pw = Process(target=write, args=(q,))
        pr = Process(target=read, args=(q,))
        # 启动子进程pw，写入:
        pw.start()
        # 启动子进程pr，读取:
        pr.start()
        # 等待pw结束:
        pw.join()
        # pr进程里是死循环，无法等待其结束，只能强行终止:
        pr.terminate()
    ```
2. multithreading  
    threading
    ```
    import time, threading
    # 新线程执行的代码:
    def loop():
        print('thread %s is running...' % threading.current_thread().name)
        n = 0
        while n < 5:
            n = n + 1
            print('thread %s >>> %s' % (threading.current_thread().name, n))
            time.sleep(1)
        print('thread %s ended.' % threading.current_thread().name)
    print('thread %s is running...' % threading.current_thread().name)
    t = threading.Thread(target=loop, name='LoopThread') # create thread
    t.start() # start thread
    t.join() # wait untill all threads done
    print('thread %s ended.' % threading.current_thread().name)
    ```
    lock: threads run crossly
    ```
    import time, threading
    # 假定这是你的银行存款:
    balance = 0
    lock = threading.Lock()
    def change_it(n):
        # 先存后取，结果应该为0:
        global balance
        balance = balance + n
        balance = balance - n
    def run_thread(n):
        for i in range(1000000):
            lock.acquire()
            try:
                change_it(n)
            finally:
                lock.release()
    t1 = threading.Thread(target=run_thread, args=(5,))
    t2 = threading.Thread(target=run_thread, args=(8,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(balance)
    ```
    multi-core CPU: python can't use all cores for multithread: Global Interpreter Lock (GIL)
    ```
    import threading, multiprocessing
    def loop():
        x = 0
        while True:
            x = x ^ 1
    for i in range(multiprocessing.cpu_count()):
        t = threading.Thread(target=loop)
        t.start()
    # have a look at CPU in task manager
    ```
3. ThreadLocal # simplify parameter calling, avoid editing global var
    ```
    import threading   
    # 创建全局ThreadLocal对象:
    local_school = threading.local()
    def process_student():
        # 获取当前线程关联的student:
        std = local_school.student
        print('Hello, %s (in %s)' % (std, threading.current_thread().name))
    def process_thread(name):
        # 绑定ThreadLocal的student:
        local_school.student = name
        process_student()
    t1 = threading.Thread(target= process_thread, args=('Alice',), name='Thread-A')
    t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    ```
4. process vs thread   
    multi-process: stable (each subprocess is independent) | large creation cost   
    multi-thread: faster | all threads share the same process memery   
    computation-dense type: rely on CPU -- C/C++ > python   
    IO-dense type: not rely on CPU -- script language > C/C++   
    asynchronous IO: CPU doesn't wait IO    
5. distributed processing  
    multiprocessing.managers: distribute tasks to different PCs
    ```
    # task_master.py
    import random,time,queue
    from multiprocessing.managers import BaseManager

    # queue1: send task
    task_queue = queue.Queue()
    # queue2: receive task
    result_queue = queue.Queue()

    # 从BaseManager继承的QueueManager:
    class QueueManager(BaseManager):
        pass
    # 把两个Queue都注册到网络上, callable参数关联了Queue对象:
    QueueManager.register('get_task_queue', callable=lambda: task_queue)
    QueueManager.register('get_result_queue', callable=lambda: result_queue)
    # 绑定端口5000, 设置验证码'abc':
    manager = QueueManager(address=('', 5000), authkey=b'abc')

    # 启动Queue:
    manager.start()
    # 获得通过网络访问的Queue对象:
    task = manager.get_task_queue()
    result = manager.get_result_queue()
    # 放几个任务进去:
    for i in range(10):
        n = random.randint(0, 10000)
        print('Put task %d...' % n)
        task.put(n)
    # 从result队列读取结果:
    print('Try get results...')
    for i in range(10):
        r = result.get(timeout=10)
        print('Result: %s' % r)
    # 关闭:
    manager.shutdown()
    print('master exit.')
    
    
    # task_worker.py  (on another PC)
    import time, sys, queue
    from multiprocessing.managers import BaseManager

    # 创建类似的QueueManager:
    class QueueManager(BaseManager):
        pass
    # 由于这个QueueManager只从网络上获取Queue，所以注册时只提供名字:
    QueueManager.register('get_task_queue')
    QueueManager.register('get_result_queue')

    # 连接到服务器，也就是运行task_master.py的机器:
    server_addr = '127.0.0.1'
    print('Connect to server %s...' % server_addr)
    # 端口和验证码注意保持与task_master.py设置的完全一致:
    m = QueueManager(address=(server_addr, 5000), authkey=b'abc')
    # 从网络连接:
    m.connect()
    # 获取Queue的对象:
    task = m.get_task_queue()
    result = m.get_result_queue()

    # 从task队列取任务,并把结果写入result队列:
    for i in range(10):
        try:
            n = task.get(timeout=1)
            print('run task %d * %d...' % (n, n))
            r = '%d * %d = %d' % (n, n, n*n)
            time.sleep(1)
            result.put(r)
        except Queue.Empty:
            print('task queue is empty.')
    # 处理结束:
    print('worker exit.')

    # usage: python task_master.py --> (another process) python task_worker.py
    ```

## 11: regular expression  
1. basics  
    `r'\d{3}\s+\d{3,8}'`: '\d' number  '\w' number or letter  '\s' space  {n} n elements  '+' more  
    special characters '\-' '\@'  
2. character range  
    `r'[0-9a-zA-Z\_]+'` # at least one letter or number or underline  
3. re module  
    `m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')` # m.group(1)=='010'  
    `re.split(r'[\s\,\;]+', 'a,b;; c  d')` # 'a','b','c','d'  
    `re.match(r'^(\d+?)(0*)$', '102300').groups()` # avoid greed match: '1023','00'  
    `re_tel = re.compile(r'^(\d{3})-(\d{3,8})$')` # re_tel.match('010-12345').groups()  
    
   
...

...

(To be continued.)






