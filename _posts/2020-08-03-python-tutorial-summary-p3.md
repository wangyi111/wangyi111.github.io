---
title: 'A summary of python tutorial (P3)'
permalink: /posts/2020/08/python-tutorial-summary-p3/
categories: programming
tags:
  - python
  - tutorial
toc: true
toc_label: "CONTENT"
---

Learning a programming language needs continuous practice and also, frequent review. To make it convenient to look back and check the basics, I write this summary based on [Xuefeng Liao's python tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400).

**note**: Some parts may be too brief to understand without previous knowledge, please see [full tutorial](https://www.liaoxuefeng.com/wiki/1016959663602400) for an addition.

**note**: This part is mostly about web developping modules because of the original author's work field. I've forgot already most of the modules here so far, which is normal. If you feel stuck here, that's normal too. Please look into details of specific modules when needed.

# Part 3: Python Modules

## 12: common built-in modules
1. datetime  
    `from datetime import datetime, timedelta, timezone`  
    `now = datetime.now()`: current time  
    `dt = datetime(2020, 7, 27, 11, 20)`: 2020-07-27 11:20:00  
    `ts = dt.timestamp()`: start: 1970-1-1 00:00:00 UTC+0:00  
    `datetime.fromtimestamp(ts)`: back to datetime  
    `datetime.utcfromtimestamp(ts)`: to utc time  
    `cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')`: str to datetime  
    `tstr = cday.strftime('%a, %b %d %H:%M')`: datetime to str  
    `dt + timedelta(hours=10)`: plus 10 hours  
    timezone tranformation:  
    ```
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc) # utc0:00
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8))) # transfer to Beijing time utc+8:00
    tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9))) # BJ to Tokyo time  
    ```
2. collections (a set of useful modules)  
    (1) namedtuple  
    `Point = collections.namedtuple('Point',['x','y'])` # p = Point(1,2)  p.x==1 p.y==2  
    (2) deque  
    `q = collections.deque(['a','b','c'])` # q.appendleft('z')  q.popleft()   more efficient  
    (3) defaultdict  
    `dd = defaultdict(lambda: 'N/A')` # dd['key'] == N/A  
    (4) OrderedDict  
    `od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])` # ordered  
    (5) ChainMap  
    ```
    # example: 参数的优先级查找
    from collections import ChainMap
    import os, argparse

    # 构造缺省参数:
    defaults = {
        'color': 'red',
        'user': 'guest'
    }

    # 构造命令行参数:
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user')
    parser.add_argument('-c', '--color')
    namespace = parser.parse_args()
    command_line_args = { k: v for k, v in vars(namespace).items() if v }

    # 组合成ChainMap:
    combined = ChainMap(command_line_args, os.environ, defaults)

    # 打印参数:
    print('color=%s' % combined['color'])
    print('user=%s' % combined['user'])
    ```
    (6) Counter  
    ```
    c = Counter()
    for ch in 'hello world!':
        c[ch] = c[ch] + 1
    print(c)
    c.update('hello love!')
    print(c)
    ```
3. base64  # binary to string  
    3 bytes (24 bit) bianry --> 4 bytes text  
    `base64.b64encode(b'binary\x00string')`  
    `base64.b64decode(b'YmluYXJ5AHN0cmluZw==')` # add '=' in the end for compensation  
4. struct: deal with bytes  
    ```
    import struct
    struct.pack('>I',10240099)
    struct.unpack('>IH',b'\xf0\xf0\xf0\xf0\x80\x80') # I:4-bytes unsigned int  H:2-bytes unsigned int  
    # example: windows bmp file
    s = b'\x42\x4d\x38\x8c\x0a\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x80\x02\x00\x00\x68\x01\x00\x00\x01\x00\x18\x00' # windows bmp file 
    # 两个字节：'BM'表示Windows位图，'BA'表示OS/2位图； 一个4字节整数：表示位图大小； 一个4字节整数：保留位，始终为0； 一个4字节整数：实际图像的偏移量； 一个4字节整数：Header的字节数； 一个4字节整数：图像宽度； 一个4字节整数：图像高度； 一个2字节整数：始终为1； 一个2字节整数：颜色数
    struct.unpack('<ccIIIIIIHH', s)
    ```
5. hashlib  
    ```
    # 通过一个函数，把任意长度的数据转换为一个长度固定的数据串
    import hashlib
    md5 = hashlib.md5()  # other algorithms: sha1, sha256, sha512
    md5.update('how to use md5 in '.encode('utf-8'))
    md5.update('python hashlib?'.encode('utf-8'))
    print(md5.hexdigest())
    ```
6. hmac: Keyed-Hashing for Message Authentication
    ```
    # example: login authentication
    # -*- coding: utf-8 -*-
    import hmac, random

    def hmac_md5(key, s):
        return hmac.new(key.encode('utf-8'), s.encode('utf-8'), 'MD5').hexdigest()

    class User(object):
        def __init__(self, username, password):
            self.username = username
            self.key = ''.join([chr(random.randint(48, 122)) for i in range(20)])
            self.password = hmac_md5(self.key, password)

    db = {
        'michael': User('michael', '123456'),
        'bob': User('bob', 'abc999'),
        'alice': User('alice', 'alice2008')
    }

    def login(username, password):
        user = db[username]
        return user.password == hmac_md5(user.key, password)
    
    assert login('michael', '123456')
    ```
7. itertools  
    ```
    import itertools
    na = itertools.count(1) # iterator 1,2,3,...
    cs = itertools.cycle('ABC') # ABCABCABC...
    cs = itertools.repeat('A',2) # AA
    ns = itertools.takewhile(lambda x: x <= 10, natuals) # 1,2,3,..,10
    itertools.chain('ABC','DEF') # ABCDEF
    for key, group in itertools.groupby('AaaBBbcCAAa', lambda c: c.upper()):
        # dict: {'A':['A','a','a'],'B':[B,B,b],...}
        print(key, list(group))
    ```
8. contextlib  
    traditional context management: to use 'with...'
    ```
    class Query(object):
        def __init__(self, name):
            self.name = name
        def __enter__(self): #
            print('Begin')
            return self
        def __exit__(self, exc_type, exc_value, traceback): #
            if exc_type:
                print('Error')
            else:
                print('End')
        def query(self):
            print('Query info about %s...' % self.name)

    with Query('Bob') as q:
        q.query()
    ```
    @contextmanager
    ```
    class Query(object):
        def __init__(self, name):
            self.name = name
        def query(self):
            print('Query info about %s...' % self.name)
    @contextmanager
    def create_query(name):
        print('Begin')
        q = Query(name)
        yield q
        print('End')
    
    with create_query('Bob') as q:
        q.query()
    ```
    @closing
    ```
    from contextlib import closing
    from urllib.request import urlopen
    with closing(urlopen('https://www.python.org')) as page:
        for line in page:
            print(line) 
    ```
9. urllib: deal with urls    
    ```
    # example: simulate login to weibo
    from urllib import request, parse
    print('Login to weibo.cn...')
    email = input('Email: ')
    passwd = input('Password: ')
    login_data = parse.urlencode([
        ('username', email),
        ('password', passwd),
        ('entry', 'mweibo'),
        ('client_id', ''),
        ('savestate', '1'),
        ('ec', ''),
        ('pagerefer', 'https://passport.weibo.cn/signin/welcome?entry=mweibo&    r=http%3A%2F%2Fm.weibo.cn%2F')
    ])

    req = request.Request('https://passport.weibo.cn/sso/login')
    req.add_header('Origin', 'https://passport.weibo.cn')
    req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
    req.add_header('Referer', 'https://passport.weibo.cn/signin/login?entry=mweibo&res=wel&wm=3349&r=http%3A%2F%2Fm.weibo.cn%2F')

    with request.urlopen(req, data=login_data.encode('utf-8')) as f:
        print('Status:', f.status, f.reason)
        for k, v in f.getheaders():
            print('%s: %s' % (k, v))
        print('Data:', f.read().decode('utf-8'))
    ```
10. XML: DOM vs SAX  
    ```
    from xml.parsers.expat import ParserCreate
    #利用SAX解析XML文档牵涉到两个部分: 解析器和事件处理器
    #解析器负责读取XML文档，并向事件处理器发送事件，如元素开始跟元素结束事件。
    #而事件处理器则负责对事件作出响应，对传递的XML数据进行处理
    class DefualtSaxHandler(object):
        def start_element(self,name,attrs):
            print('sax:start_elment: %s,attrs: %s'%(name,str(attrs)))
            #name表示节点名称，attrs表示节点属性（字典）
        def end_element(self,name):
            print('sax:end_element: %s'%name)
        def char_data(self,text):
            print('sax:char_data: %s'%text)
            #text表示节点数据
    xml=r'''<?xml version="1.0"?>
    <ol>
        <li><a href="/python">Python</a></li>
        <li><a href="/ruby">Ruby</a></li>
    </ol>
    '''

    #处理器实例
    handler=DefualtSaxHandler()
    #解析器实例
    parser=ParserCreate()

    #下面3为解析器设置自定义的回调函数
    #回调函数的概念，请搜索知乎，见1.9K赞的答案
    parser.StartElementHandler=handler.start_element
    parser.EndElementHandler=handler.end_element
    parser.CharacterDataHandler=handler.char_data
    #开始解析XML
    parser.Parse(xml)
    #然后就是等待expat解析，
    #一旦expat解析器遇到xml的 元素开始，元素结束，元素值 事件时
    #会回分别调用start_element, end_element, char_data函数

    #关于XMLParser Objects的方法介绍下
    #详见python文档：xml.parsers.expat
    #xmlparser.StartElementHandler(name, attributes)
    #遇到XML开始标签时调用，name是标签的名字，attrs是标签的属性值字典
    #xmlparser.EndElementHandler(name)
    #遇到XML结束标签时调用。
    #xmlparser.CharacterDataHandler(data) 
    #调用时机：
    #从行开始，遇到标签之前，存在字符，content 的值为这些字符串。
    #从一个标签，遇到下一个标签之前， 存在字符，content 的值为这些字符串。
    #从一个标签，遇到行结束符之前，存在字符，content 的值为这些字符串。
    #标签可以是开始标签，也可以是结束标签。

    #为了方便理解，我已经在下面还原来解析过程，
    #标出何时调用，分别用S：表示开始；E：表示结束；D：表示data
    ```
11. HTMLParser  
    ```
    # example
    from html.parser import HTMLParser
    from html.entities import name2codepoint

    class MyHTMLParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            print('<%s>' % tag)
        def handle_endtag(self, tag):
            print('</%s>' % tag)
        def handle_startendtag(self, tag, attrs):
            print('<%s/>' % tag)
        def handle_data(self, data):
            print(data)
        def handle_comment(self, data):
            print('<!--', data, '-->')
        def handle_entityref(self, name):
            print('&%s;' % name)
        def handle_charref(self, name):
            print('&#%s;' % name)

    parser = MyHTMLParser()
    parser.feed('''<html>
    <head></head>
    <body>
    <!-- test html parser -->
        <p>Some <a href=\"#\">html</a> HTML&nbsp;tutorial...<br>END</p>
    </body></html>''') # feed can be called multiple times
    ```

## 13: third-party modules
1. Pillow: image processing  
    ```
    from PIL import Image
    # 打开一个jpg图像文件，注意是当前路径:
    im = Image.open('test.jpg')
    # 获得图像尺寸:
    w, h = im.size
    print('Original image size: %sx%s' % (w, h))
    # 缩放到50%:
    im.thumbnail((w//2, h//2))
    print('Resize image to: %sx%s' % (w//2, h//2))
    # blur
    im2 = im.filter(ImageFilter.BLUR)
    # 把缩放blur后的图像用jpeg格式保存:
    im2.save('thumbnail.jpg', 'jpeg')
    ```
    ImageDraw # draw image
    ```
    # example: 生成字母验证码图片
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import random
    # 随机字母:
    def rndChar():
        return chr(random.randint(65, 90))
    # 随机颜色1:
    def rndColor():
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
    # 随机颜色2:
    def rndColor2():
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))
    # 240 x 60:
    width = 60 * 4
    height = 60
    image = Image.new('RGB', (width, height), (255, 255, 255))
    # 创建Font对象:
    font = ImageFont.truetype('Arial.ttf', 36)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 填充每个像素:
    for x in range(width):
        for y in range(height):
            draw.point((x, y), fill=rndColor())
    # 输出文字:
    for t in range(4):
        draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
    # 模糊:
    image = image.filter(ImageFilter.BLUR)
    image.save('code.jpg', 'jpeg')
    ```
2. requests: visit web resource
    `r = requests.get('https://www.douban.com/',params={},headers={})` # r.text   
    `r = requests.post('https://accounts.douban.com/login',data={'form_email': 'abc@example.com', 'form_password': '123456'})`   
    ...   
3. chardet: detect encoding type  
    `chardet.detect(b'Hello, world!')`
4. psutil: process and system utilities  
    ```
    # examples
    import psutil
    psutil.cpu_count() # CPU逻辑数量
    psutil.cpu_count(logical=False) # CPU物理核心
    # 2说明是双核超线程, 4则是4核非超线程
    psutil.cpu_times() # CPU的用户／系统／空闲时间
    for x in range(10):
        print(psutil.cpu_percent(interval=1, percpu=True)) # CPU使用率，每秒刷新一次，累计10次
    psutil.virtual_memory()
    psutil.swap_memory()
    psutil.disk_partitions() # 磁盘分区信息
    psutil.disk_usage('/') # 磁盘使用情况
    psutil.disk_io_counters() # 磁盘IO
    psutil.net_io_counters() # 获取网络读写字节／包的个数
    psutil.net_if_addrs() # 获取网络接口信息
    psutil.net_if_stats() # 获取网络接口状态
    psutil.net_connections() # network connection info, need administration authority 'sudo python3'
    psutil.pids() # 所有进程ID
    p = psutil.Process(3776) # 获取指定进程ID=3776，其实就是当前Python交互环境
    # p.name() p.exe() p.cwd() ...
    ```
5. virtualenv: virtual python environment, each project works in its env  
    see virtualenv documentation for details  

## 14: graph user interface  
1. TKinter  
    ```
    from tkinter import *
    import tkinter.messagebox as messagebox
    class Application(Frame):
        def __init__(self,master=None):
            #super(Application,self).__init__(master)
            Frame.__init__(self, master)
            self.pack()
            self.createWidgets()
        def createWidgets(self):
            self.nameInput = Entry(self)
            self.nameInput.pack()
            self.alterButton = Button(self,text='Hello',command=self.hello)
            self.alterButton.pack()
        def hello(self):
            name = self.nameInput.get() or 'world'
            messagebox.showinfo('Message','Hello, %s' % name)

    app = Application()
    app.master.title('Hello world')
    app.mainloop()
    ```
2. Turtle Graphics  
    ```
    # example: draw a parting tree
    from turtle import *

    # 设置色彩模式是RGB:
    colormode(255)

    lt(90)

    lv = 14
    l = 120
    s = 45

    width(lv)

    # 初始化RGB颜色:
    r = 0
    g = 0
    b = 0
    pencolor(r, g, b)

    penup()
    bk(l)
    pendown()
    fd(l)

    def draw_tree(l, level):
        global r, g, b
        # save the current pen width
        w = width()

        # narrow the pen width
        width(w * 3.0 / 4.0)
        # set color:
        r = r + 1
        g = g + 2
        b = b + 3
        pencolor(r % 200, g % 200, b % 200)

        l = 3.0 / 4.0 * l

        lt(s)
        fd(l)

        if level < lv:
            draw_tree(l, level + 1)
        bk(l)
        rt(2 * s)
        fd(l)

        if level < lv:
            draw_tree(l, level + 1)
        bk(l)
        lt(s)

        # restore the previous pen width
        width(w)

    speed("fastest")

    draw_tree(l, 4)

    done()
    ```

# 15: web programming  
1. TCP/IP  
2. TCP programming # relible connection via stream data  
    client programming:  
    ```
    # example 1: visit sina.com and save the home page
    import socket
    import ssl

    # create socket for connection
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # ipv4 , stream tcp
    # create connection
    s = ssl.wrap_socket(socket.socket())
    s.connect(('www.sina.com.cn', 443))
    # s.connect(('www.sina.com.cn',80)) # port 80 standard for web service
    s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')

    # receive data
    buffer = []
    while True:
        d = s.recv(1024) # 1024 bytes each time
        if d:
            buffer.append(d)
        else:
            break
    data = b''.join(buffer)
    s.close()
    # get content
    header, html = data.split(b'\r\n\r\n',1)
    print(header.decode('utf-8'))
    with open('sina.html', 'wb') as f:
        f.write(html)
    ```
    server programming:  
    ```
    # example 2: server and client
    # server.py
    import socket
    import threading
    import time
    # create socket
    # 一个Socket依赖4项：服务器地址、服务器端口、客户端地址、客户端端口
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 监听端口:
    s.bind(('127.0.0.1', 9999)) # local host IP , costom port 
    s.listen(5) # listen to at most 5 ports
    print('Waiting for connection...')

    def tcplink(sock, addr):
        print('Accept new connection from %s:%s...' % addr)
        sock.send(b'Welcome!')
        while True:
            data = sock.recv(1024)
            time.sleep(1)
            if not data or data.decode('utf-8') == 'exit':
                break
            sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
        sock.close()
        print('Connection from %s:%s closed.' % addr)

    while True:
        # 接受一个新连接:
        sock, addr = s.accept()
        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=tcplink, args=(sock, addr))
        t.start()
    
    # client.py 
    import socket
    # create socket for connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 建立连接:
    s.connect(('127.0.0.1', 9999))
    # 接收欢迎消息:
    print(s.recv(1024).decode('utf-8'))
    for data in [b'Michael', b'Tracy', b'Sarah']:
        # 发送数据:
        s.send(data)
        print(s.recv(1024).decode('utf-8'))
    s.send(b'exit')
    s.close()
    
    # usage:
    # open two seperate command line window
    # cmd1:
    $ python server.py
    # cmd2:
    $ python client.py
    ```
3. UDP programming: no connection, unstable but faster
    ```
    # example:
    # server.py

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # SOCK_DGRAM: UDP
    # 绑定端口:
    s.bind(('127.0.0.1', 9999))
    print('Bind UDP on 9999...')
    while True:
        # 接收数据:
        data, addr = s.recvfrom(1024)
        print('Received from %s:%s.' % addr)
        s.sendto(b'Hello, %s!' % data, addr)
    
    # client.py 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for data in [b'Michael', b'Tracy', b'Sarah']:
        # 发送数据:
        s.sendto(data, ('127.0.0.1', 9999))
        # 接收数据:
        print(s.recv(1024).decode('utf-8'))
    s.close()
    ```

## 16: email   
0. email process:  
    `sender --> MUA(Mail User Agent) --> MTA(Mail Transfer Agent) --> MTA --> MDA(Mail Delivery Agent) <-- MUA <-- receiver`   
    send email: SMTP (Simple Mail Transfer Protocol)  
    receive email: POP3 (Post Office Protocol); IMAP (Internet Message Access Protocol)  
1. send email: SMTP  
    ```
    # example: send a plain text email
    from email import encoders
    from email.header import Header
    from email.mime.text import MIMEText
    from email.utils import parseaddr, formataddr

    import smtplib
        
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    from_addr = input('From: ')
    password = input('Password: ')
    to_addr = input('To: ')
    smtp_server = input('SMTP server: ')

    msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
    msg['From'] = _format_addr('Python爱好者 <%s>' % from_addr)
    msg['To'] = _format_addr('管理员 <%s>' % to_addr)
    msg['Subject'] = Header('来自SMTP的问候……', 'utf-8').encode()
        
    server = smtplib.SMTP(smtp_server, 25)
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()
    ```
2. receive email: POP3  
    ```
    # example: receive email
    # step 1: 用poplib把邮件的原始文本下载到本地
    import poplib

    # 输入邮件地址, 口令和POP3服务器地址:
    email = input('Email: ')
    password = input('Password: ')
    pop3_server = input('POP3 server: ')

    # 连接到POP3服务器:
    server = poplib.POP3(pop3_server)
    # 可以打开或关闭调试信息:
    server.set_debuglevel(1)
    # 可选:打印POP3服务器的欢迎文字:
    print(server.getwelcome().decode('utf-8'))

    # 身份认证:
    server.user(email)
    server.pass_(password)

    # stat()返回邮件数量和占用空间:
    print('Messages: %s. Size: %s' % server.stat())
    # list()返回所有邮件的编号:
    resp, mails, octets = server.list()
    # 可以查看返回的列表类似[b'1 82923', b'2 2184', ...]
    print(mails)

    # 获取最新一封邮件, 注意索引号从1开始:
    index = len(mails)
    resp, lines, octets = server.retr(index)

    # lines存储了邮件的原始文本的每一行,
    # 可以获得整个邮件的原始文本:
    msg_content = b'\r\n'.join(lines).decode('utf-8')
    # 稍后解析出邮件:
    msg = Parser().parsestr(msg_content)

    # 可以根据邮件索引号直接从服务器删除邮件:
    # server.dele(index)
    # 关闭连接:
    server.quit()
    
    # step 2: 用email解析原始文本
    from email.parser import Parser
    from email.header import decode_header
    from email.utils import parseaddr
    import poplib
    msg = Parser().parsestr(msg_content) # parser email as Message object
    # print structure of Message
    def print_info(msg, indent=0):
        if indent == 0:
            for header in ['From', 'To', 'Subject']:
                value = msg.get(header, '')
                if value:
                    if header=='Subject':
                        value = decode_str(value)
                    else:
                        hdr, addr = parseaddr(value)
                        name = decode_str(hdr)
                        value = u'%s <%s>' % (name, addr)
                print('%s%s: %s' % ('  ' * indent, header, value))
        if (msg.is_multipart()):
            parts = msg.get_payload()
            for n, part in enumerate(parts):
                print('%spart %s' % ('  ' * indent, n))
                print('%s--------------------' % ('  ' * indent))
                print_info(part, indent + 1)
        else:
            content_type = msg.get_content_type()
            if content_type=='text/plain' or content_type=='text/html':
                content = msg.get_payload(decode=True)
                charset = guess_charset(msg)
                if charset:
                    content = content.decode(charset)
                print('%sText: %s' % ('  ' * indent, content + '...'))
            else:
                print('%sAttachment: %s' % ('  ' * indent, content_type))
    # decode title or name 
    def decode_str(s):
        value, charset = decode_header(s)[0]
        if charset:
            value = value.decode(charset)
        return value
    # examine encoding of text
    def guess_charset(msg):
        charset = msg.get_charset()
        if charset is None:
            content_type = msg.get('Content-Type', '').lower()
            pos = content_type.find('charset=')
            if pos >= 0:
                charset = content_type[pos + 8:].strip()
        return charset
    ```

...

...

(To be continued.)

    






    
    



