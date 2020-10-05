---
title: 'C++标准库常用数据结构和算法（转）'
permalink: /posts/2020/08/c++-std-library/
categories: programming
tags:
  - C++
  - stl
toc: true
toc_label: "CONTENT"
---

本文整理了C++标准库STL常用的数据结构和算法，统一使用命名空间std，原文参考https://blog.ailemon.me/2019/03/29/cpp-stl-library-summary/。

```C++
using namespace std;
```

## Vector

vector类似于平时使用的数组类型，只不过，封装了很多常用的函数，并且长度是可变的。
```C++
#include <vector>
```
示例：
```C++
vector<int> a;

//向vector的末尾添加元素
for(int i=1;i<10;i++)
        a.push_back(i);

//输出首元素
cout << a.front()<<endl;

//输出尾元素
cout << a.back()<<endl;

//利用指针和begin函数和end函数进行遍历
vector<int>::iterator it;
for(it=a.begin();it!=a.end();it++)
        cout << *it << " ";
cout <<endl;

//使用size函数进行遍历
int len=a.size();
for(int i=0;i<len;i++)
        cout << a[i] << " ";
cout <<endl;

//删除队尾元素
a.pop_back();
cout << a.back()<<endl;

//通过erase函数删除第二个元素
it=a.begin()+1;

a.erase(it);//删除的是2
len=a.size();
for(int i=0;i<len;i++)
        cout << a[i] << " ";
cout <<endl;

//通过erase函数删除前两个元素
vector<int>::iterator it2;
it=a.begin();
it2=a.begin()+2;//删除的是1，3（注意这里不删除it2指向的那个位置）
a.erase(it,it2);//就是说删除[it,it2)的部分
len=a.size();
for(int i=0;i<len;i++)
        cout << a[i] << " ";
cout <<endl;

//清空整个vector
a.clear();

//判断是否为空
if(a.empty()) cout << "这个vector现在是空的了" <<endl;

//向指针指向的元素的前面插一个元素进去
for(int i=1;i<10;i++)
        a.push_back(i);
it=a.begin();
a.insert(it,-1);//这样的插入是可以返回插入元素的指针的，而其他的插入方式就不行
len=a.size();
for(int i=0;i<len;i++)
        cout << a[i] << " ";
cout <<endl;

//向指针指向的元素的前面插4个相同的元素进去
it=a.begin()+1;
a.insert(it,4,100);
len=a.size();
for(int i=0;i<len;i++)
        cout << a[i] << " ";
cout <<endl;

//使用sort函数
sort(a.begin(),a.end());

//完整交换两个不同的vector
vector<int> a2;
for(int i=1;i<10;i++)
        a2.push_back(i);
a.swap(a2);
len=a.size();
for(int i=0;i<len;i++)
        cout << a[i] << " ";
cout <<endl;
len=a2.size();
for(int i=0;i<len;i++)
        cout << a2[i] << " ";
cout <<endl;
```

## Stack

```C++
#include <stack>
```

stack 模板类需要两个模板参数，一个是元素类型，一个容器类型，但只有元素类型是必要的，在不指定容器类型时，默认的容器类型为deque。

定义stack对象: `stack s1;`

stack 的基本操作有：
* 入栈，如例：s.push(x);
* 出栈，如例：s.pop();注意，出栈操作只是删除栈顶元素，并不返回该元素。
* 访问栈顶，如例：s.top()
* 判断栈空，如例：s.empty()，当栈空时，返回true。
* 访问栈中的元素个数，如例：s.size()。

示例：
```C++
stack<int> a;
//入栈操作
a.push(1);
a.push(2);
a.push(3);

//取栈顶元素输出
cout << a.top() << endl;

//删除栈顶元素
a.pop();

//再次取栈顶元素
cout << a.top() << endl;

//判断栈是否为空
if(!a.empty()) cout << "栈不为空" << endl;
```

## Queue

与stack 模板类很相似，queue 模板类也需要两个模板参数，一个是元素类型，一个容器类型，元素类型是必要的，容器类型是可选的，默认为deque 类型。

```C++
#include <queue>
```

定义stack对象: `stack s1;`

queue 的基本操作有：
* 入队，如例：q.push(x); 将x 接到队列的末端。
* 出队，如例：q.pop(); 弹出队列的第一个元素，注意，并不会返回被弹出元素的值。
* 访问队首元素，如例：q.front()，即最早被压入队列的元素。
* 访问队尾元素，如例：q.back()，即最后被压入队列的元素。
* 判断队列空，如例：q.empty()，当队列空时，返回true。
* 访问队列中的元素个数，如例：q.size()

```C++
queue<int> a;
//插入元素
a.push(2);
a.push(1);
a.push(3);
a.push(4);

//访问队列首元素,注意这里top是用不成的
cout << a.front()<<endl;

//访问队列尾元素
cout << a.back()<<endl;

//删除首元素，但是他不返回值
a.pop();
cout << a.front()<<endl;

//判断队列是否为空
if(!a.empty()) cout << "队列不为空" <<endl;

//输出队列中的元素个数
cout << a.size() << endl;
```

## Priority_queue

优先队列和队列中的操作指令基本上类似，不过，优先队列是通过堆来实现的。

```C++
#include <queue>
```

优先队列是默认大的数在前面，若是想定义小的数在前面只需要像下面这样定义即可：
```C++
priority_queue<int,vector<int>,greater<int>> a;
```
示例：
```C++
priority_queue<int> a;
//插入元素
a.push(2);
a.push(1);
a.push(3);
a.push(4);

//访问队列首元素,注意这里不是像队列一样使用front和back来访问首元素和尾元素
cout << a.top()<<endl;

//删除首元素，但是他不返回值
a.pop();
cout << a.top
()<<endl;

//判断队列是否为空
if(!a.empty()) cout << "队列不为空" <<endl;

//输出队列中的元素个数
cout << a.size() << endl;
```

## Set

```C++
#include <set>
```
示例：
```C++
set<int> s;
s.insert(1);
s.insert(3);
s.insert(5);
//查找元素
set<int>::iterator it;
it=s.find(1);
if(it==s.end()) puts("not found");
else puts("found");    //输出found

it=s.find(2);
if(it==s.end()) puts("not found");
else puts("found");    //输出not found

//删除元素
s.erase(3);

//其他的查找元素的方法
if(s.count(3)!=0) puts("found");
else puts("not found");//输出 not found

//遍历所有元素
for(it=s.begin();it!=s.end();it++){
        printf("%d\n",*it);
}
```

## Map

```C++
#include <map>
```
示例：
```C++
map<int,const char *> m;

//插入元素
m.insert(make_pair(1,"ONE"));
m.insert(make_pair(10,"TEN"));  //标准的写法
m[100]="HUNDRED";               //另一种插入元素的写法

//查找元素
map<int,const char*>::iterator it;
it=m.find(1);
puts(it->second);   //输出ONE

it=m.find(2);
if(it==m.end()) puts("not found");
else puts("found");      //输出not found

puts(m[10]);             //其他的输出方式

//删除元素
m.erase(10);

//遍历一遍所有的元素
for(it = m.begin();it!=m.end();it++)
        printf("%d: %s",it->first,it->second);
```

## Pair

1，初始化：
```C++
pair<string, string> anon;//调用默认构造函数来初始化
pair<string, int> word_count;
pair<string, vector<int>> line;

pair<string, string> author("jim", "weshon"); //定义时提供初始化
typedef pair<string, string> Author;
Author product("marcel", "Product");
```
2，pair对象的操作，支持== 、< ，first、second成员的访问：
```C++
string firstbook;
if (author.first == "jim" && author.second == "weshon")
{
    firstbook = "stephon hero"; //
}
```
3，生成新的pair对象：
```C++
pair<string, string> next_author;
string first, second;
while (cin >> first >> second)
{
    next_author = make_pair(first, second);
    //pair<string, string> next_author(first, second); //与上面创建类似
}
```

## Algorithm

```C++
#include <algorithm>    // std::sort
```
这里介绍sort示例：
bool myfunction (int i,int j) { return (i<j); }

  // using function as comp
  std::sort (myvector.begin()+4, myvector.end(), myfunction); // 12 32 45 71(26 33 53 80)



struct myclass {
  bool operator() (int i,int j) { return (i<j);}
} myobject;

  // using object as comp
  std::sort (myvector.begin(), myvector.end(), myobject);     //(12 26 32 33 45 53 71 80)
```

