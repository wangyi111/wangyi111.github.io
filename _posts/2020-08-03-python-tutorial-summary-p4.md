---
title: 'A summary of python tutorial (P4)'
permalink: /posts/2020/08/python-tutorial-summary-p4/
categories: programming
tags:
  - python
  - tutorial
toc: true
toc_label: "CONTENT"
---

Learning a programming language needs continuous practice and also, frequent review. To make it convenient to look back and check the basics, I write this summary based on [Xuefeng Liao's python tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400).

**note**: Some parts may be too brief to understand without previous knowledge, please see [full tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400) for an addition.

**note**: This part is about web development, which I didn't even understand a lot when first learning the tutorial. If you find it difficult, it's not because you are not smart, you are just as clever as I was. Please look into other more specific and novice-friendly tutorials for systematic learning.

# Part 4: web development

## 17: Database
1. SQLite: light & embeddable, desktop/mobile oriented
    ```
    # 导入SQLite驱动:
    >>> import sqlite3
    # 连接到SQLite数据库
    # 数据库文件是test.db
    # 如果文件不存在，会自动在当前目录创建:
    >>> conn = sqlite3.connect('test.db')
    # 创建一个Cursor:
    >>> cursor = conn.cursor()
    # 执行一条SQL语句，创建user表:
    >>> cursor.execute('create table user (id varchar(20) primary key, name varchar(20))')
    <sqlite3.Cursor object at 0x10f8aa260>
    # 继续执行一条SQL语句，插入一条记录:
    >>> cursor.execute('insert into user (id, name) values (\'1\', \'Michael\')')
    <sqlite3.Cursor object at 0x10f8aa260>
    # 通过rowcount获得插入的行数:
    >>> cursor.rowcount
    1
    # 关闭Cursor:
    >>> cursor.close()
    # 提交事务:
    >>> conn.commit()
    # 关闭Connection:
    >>> conn.close()

    #############################
    # query
    >>> conn = sqlite3.connect('test.db')
    >>> cursor = conn.cursor()
    # 执行查询语句:
    >>> cursor.execute('select * from user where id=?', ('1',))
    <sqlite3.Cursor object at 0x10f8aa340>
    # 获得查询结果集:
    >>> values = cursor.fetchall()
    >>> values
    [('1', 'Michael')]
    >>> cursor.close()
    >>> conn.close()
    ```
2. MySQL: most popular, server oriented  
    ```
    # download and install MySQL
    # $ pip install mysql-connector-python --allow-external mysql-connector-python

    # 导入MySQL驱动:
    >>> import mysql.connector
    # 注意把password设为你的root口令:
    >>> conn = mysql.connector.connect(user='root', password='password', database='test')
    >>> cursor = conn.cursor()
    # 创建user表:
    >>> cursor.execute('create table user (id varchar(20) primary key, name varchar(20))')
    # 插入一行记录，注意MySQL的占位符是%s:
    >>> cursor.execute('insert into user (id, name) values (%s, %s)', ['1', 'Michael'])
    >>> cursor.rowcount
    1
    # 提交事务:
    >>> conn.commit()
    >>> cursor.close()
    # 运行查询:
    >>> cursor = conn.cursor()
    >>> cursor.execute('select * from user where id = %s', ('1',))
    >>> values = cursor.fetchall()
    >>> values
    [('1', 'Michael')]
    # 关闭Cursor和Connection:
    >>> cursor.close()
    True
    >>> conn.close()
    ```
3. SQLAlchemy: object-relational mapping (ORM--把数据库表的一行记录与一个对象互相做自动转换)
    ```
    # pip install sqlalchemy

    # 导入:
    from sqlalchemy import Column, String, create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base

    # 创建对象的基类:
    Base = declarative_base()

    # 定义User对象, 如果有多个表，就继续定义其他class
    class User(Base):
        # 表的名字:
        __tablename__ = 'user'

        # 表的结构:
        id = Column(String(20), primary_key=True)
        name = Column(String(20))

    # 初始化数据库连接: '数据库类型+数据库驱动名称://用户名:口令@机器地址:端口号/数据库名'
    engine = create_engine('mysql+mysqlconnector://root:password@localhost:3306/test') 
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)

    # 向数据库表中添加一行记录，可以视为添加一个User对象
    # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    new_user = User(id='5', name='Bob')
    # 添加到session:
    session.add(new_user)
    # 提交即保存到数据库:
    session.commit()
    # 关闭session:
    session.close()

    # 查询出来的可以不再是tuple，而是User对象
    # 创建Session:
    session = DBSession()
    # 创建Query查询，filter是where条件，最后调用one()返回唯一行，如果调用all()则返回所有行:
    user = session.query(User).filter(User.id=='5').one()
    # 打印类型和对象的name属性:
    print('type:', type(user))
    print('name:', user.name)
    # 关闭Session:
    session.close()
    ```

## 18: Web development  
0. web application  
    \[browser\] send http request 'header + (if needed) body (interaction)'  
    \[server\] receive http request, generate a html  
    \[server\] send http response (html as body) to browser  
    \[browser\] get http response, extract html and show  
1. http  

2. html  

3. WSGI # web server gateway interface  
    ```
    # convenient for http response
    # python built-in WSGI server: wsgiref
    # example:
    # hello.py: WSGI processing function --- entrance for a wep app
    def application(environ,start_response):
        # environ: a dict including all http request
        # start_response: a function sending http response
        start_response('200 OK',[('Content-Type','text/html')]) # send http response header
        body = b'<h1>Hello, %s!</h1>' % (environ['PATH_INFO'][1:] or 'web')
        return [body.encode('utf-8')] # http response body
    
    # server.py: start WSGI server, load application()
    from wsgiref.simple_server import make_server
    # 导入我们自己编写的application函数:
    from hello import application
    # 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
    httpd = make_server('', 8000, application)
    print('Serving HTTP on port 8000...')
    # 开始监听HTTP请求:
    httpd.serve_forever()

    #######################
    # usage:
    # python server.py 
    # open http://localhost:8000/yiwang
    ```
4. Web framework: 处理URL到函数的映射  
    popular framework: flask, Django, web.py, Bottle, ...  
    ```
    # example: flask
    # pip install flask

    from flask import Flask
    from flask import request

    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def home():
        return '<h1>Home</h1>'

    @app.route('/signin', methods=['GET'])
    def signin_form():
        return '''<form action="/signin" method="post">
                <p><input name="username"></p>
                <p><input name="password" type="password"></p>
                <p><button type="submit">Sign In</button></p>
                </form>'''

    @app.route('/signin', methods=['POST'])
    def signin():
        # 需要从request对象读取表单内容：
        if request.form['username']=='admin' and request.form['password']=='password':
            return '<h3>Hello, admin!</h3>'
        return '<h3>Bad username or password.</h3>'

    if __name__ == '__main__':
        app.run()
    ```
5. template: for html design
    MVC: model view controller  # isolate python and html
    ```
    # example
    # server.py
    from flask import Flask, request, render_template

    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def home():
        return render_template('home.html')

    @app.route('/signin', methods=['GET'])
    def signin_form():
        return render_template('form.html')

    @app.route('/signin', methods=['POST'])
    def signin():
        username = request.form['username']
        password = request.form['password']
        if username=='admin' and password=='password':
            return render_template('signin-ok.html', username=username)
        return render_template('form.html', message='Bad username or password', username=username)

    if __name__ == '__main__':
        app.run()
    ################################
    # home.html
    <html>
    <head>
    <title>Home</title>
    </head>
    <body>
    <h1 style="font-style:italic">Home</h1>
    </body>
    </html>

    # form.html
    <html>
    <head>
    <title>Please Sign In</title>
    </head>
    <body>
    {% if message %}
    <p style="color:red">{{ message }}</p>
    {% endif %}
    <form action="/signin" method="post">
        <legend>Please sign in:</legend>
        <p><input name="username" placeholder="Username" value="{{ username }}"></p>
        <p><input name="password" placeholder="Password" type="password"></p>
        <p><button type="submit">Sign In</button></p>
    </form>
    </body>
    </html>

    # sign-ok.html
    <html>
    <head>
    <title>Welcome, {{ username }}</title>
    </head>
    <body>
    <p>Welcome, {{ username }}!</p>
    </body>
    </html>
    ```

## 19: asynchronous IO
1. coroutine  
    ```
    # example:
    def consumer(): # generator
        r = ''
        while True:
            n = yield r
            if not n:
                return
            print('[CONSUMER] Consuming %s...' % n)
            r = '200 OK'

    def produce(c): 
        c.send(None) # start generator
        n = 0
        while n < 5: # 消息循环
            n = n + 1
            print('[PRODUCER] Producing %s...' % n)
            r = c.send(n) # switch to comsumer()
            print('[PRODUCER] Consumer return: %s' % r)
        c.close()

    c = consumer()
    produce(c)

    # 注意到consumer函数是一个generator，把一个consumer传入produce后：
    # 1. 首先调用c.send(None)启动生成器；
    # 2. 然后，一旦生产了东西，通过c.send(n)切换到consumer执行；
    # 3. consumer通过yield拿到消息，处理，又通过yield把结果传回；
    # 4. produce拿到consumer处理的结果，继续生产下一条消息；
    # 5. produce决定不生产了，通过c.close()关闭consumer，整个过程结束。
    # 整个流程无锁，由一个线程执行，produce和consumer协作完成任务，所以称为“协程”，而非线程的抢占式多任务
    ```
2. asyncio  
    ```
    # example:
    import asyncio

    @asyncio.coroutine # 把一个generator标记为coroutine类型
    def wget(host):
        print('wget %s...' % host)
        connect = asyncio.open_connection(host, 80)
        reader, writer = yield from connect ## 调用另一个coroutine
        header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
        writer.write(header.encode('utf-8'))
        yield from writer.drain() ##
        while True:
            line = yield from reader.readline() ##
            if line == b'\r\n':
                break
            print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
        # Ignore the body, close the socket
        writer.close()

    loop = asyncio.get_event_loop()
    tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    ```
3. async/await  
    `@asyncio.coroutine --> async`
    `yield from --> await`
    ```
    # @asyncio.coroutine
    async def hello():
        print("Hello world!")
        # r = yield from asyncio.sleep(1)
        r = await asyncio.sleep(1)
        print("Hello again!")
    ```
4. aiohttp: 基于asyncio实现的HTTP框架  
    ```
    import asyncio
    from aiohttp import web

    async def index(request):
        await asyncio.sleep(0.5)
        return web.Response(body=b'<h1>Index</h1>')

    async def hello(request):
        await asyncio.sleep(0.5)
        text = '<h1>hello, %s!</h1>' % request.match_info['name']
        return web.Response(body=text.encode('utf-8'))

    async def init(loop):
        app = web.Application(loop=loop)
        app.router.add_route('GET', '/', index)
        app.router.add_route('GET', '/hello/{name}', hello)
        srv = await loop.create_server(app.make_handler(), '127.0.0.1', 8000)
        print('Server started at http://127.0.0.1:8000...')
        return srv

    loop = asyncio.get_event_loop()
    loop.run_until_complete(init(loop))
    loop.run_forever()
    ```

## 20: MicroPython  

light version of python for micro controllers (small robots with raspberryPI, ... )   
I don't have money to buy an experiment sample, to be learned later :)  

   
# Congrats!   
   
Congrats for completing the tutorial! I believe you re-learned a lot even though feeling like an idiot at some point. Personally, I gave up quite a lot of times at the beginning (some parts I still have not picked up since) but finally reached here. That feeling was awesome. Let's keep moving on!

    