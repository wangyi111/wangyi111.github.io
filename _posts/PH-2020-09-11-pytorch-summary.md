---
title: 'Pytorch: Brief Summary'
permalink: /posts/2020/09/pytorch-summary/
categories: programming
tags:
  - pytorch
toc: true
toc_label: "CONTENT"
---

This is a brief summary for the implementation of pytorch, from data preparation to the usage of trained model.

## 01: Prepare Data

Pytorch通常使用Dataset和DataLoader这两个工具类来构建数据管道。Dataset定义了数据集的内容，它相当于一个类似列表的数据结构，具有确定的长度，能够用索引获取数据集中的元素;而DataLoader定义了按batch加载数据集的方法，它是一个实现了__iter__方法的可迭代对象，每次迭代输出一个batch的数据。

在绝大部分情况下，用户只需实现Dataset的__len__方法和__getitem__方法，就可以轻松构建自己的数据集，并用默认数据管道进行加载。

Dataset创建数据集常用的方法有：

* 使用 torch.utils.data.TensorDataset 根据Tensor创建数据集(numpy的array，Pandas的DataFrame需要先转换成Tensor)。
* 使用 torchvision.datasets.ImageFolder 根据图片目录创建图片数据集。
* 继承 torch.utils.data.Dataset 创建自定义数据集。

此外，还可以通过
* torch.utils.data.random_split 将一个数据集分割成多份，常用于分割训练集，验证集和测试集。
* 调用Dataset的加法运算符(+)将多个数据集合并成一个数据集。

### 1-1: 根据Tensor创建数据集

```python
import numpy as np 
import torch 
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split 

```

```python
# 根据Tensor创建数据集
from sklearn import datasets 
iris = datasets.load_iris()
ds_iris = TensorDataset(torch.tensor(iris.data),torch.tensor(iris.target))

# 分割成训练集和预测集
n_train = int(len(ds_iris)*0.8)
n_valid = len(ds_iris) - n_train
ds_train,ds_valid = random_split(ds_iris,[n_train,n_valid])

print(type(ds_iris))
print(type(ds_train))

```

```python
# 使用DataLoader加载数据集
dl_train,dl_valid = DataLoader(ds_train,batch_size = 8),DataLoader(ds_valid,batch_size = 8)

for features,labels in dl_train:
    print(features,labels)
    break
```

```python
# 演示加法运算符（`+`）的合并作用
ds_data = ds_train + ds_valid

print('len(ds_train) = ',len(ds_train))
print('len(ds_valid) = ',len(ds_valid))
print('len(ds_train+ds_valid) = ',len(ds_data))

print(type(ds_data))

```

### 1-2: 根据图片目录创建图片数据集

```python
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms,datasets 

```

```python
#演示一些常用的图片增强操作
from PIL import Image
img = Image.open('./data/cat.jpeg')
# 随机数值翻转
transforms.RandomVerticalFlip()(img)
#随机旋转
transforms.RandomRotation(45)(img)
# 定义图片增强操作

transform_train = transforms.Compose([
   transforms.RandomHorizontalFlip(), #随机水平翻转
   transforms.RandomVerticalFlip(), #随机垂直翻转
   transforms.RandomRotation(45),  #随机在45度角度内旋转
   transforms.ToTensor() #转换成张量
  ]
) 

transform_valid = transforms.Compose([
    transforms.ToTensor()
  ]
)

```

```python
# 根据图片目录创建数据集
ds_train = datasets.ImageFolder("./data/cifar2/train/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./data/cifar2/test/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())

print(ds_train.class_to_idx)

```

```
{'0_airplane': 0, '1_automobile': 1}
```

```python
# 使用DataLoader加载数据集

dl_train = DataLoader(ds_train,batch_size = 50,shuffle = True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size = 50,shuffle = True,num_workers=3)
```

```python
for features,labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break
```

```
torch.Size([50, 3, 32, 32])
torch.Size([50, 1])
```

### 1-3: 创建自定义数据集

下面通过继承Dataset类创建imdb文本分类任务的自定义数据集。

大概思路如下：首先，对训练集文本分词构建词典。然后将训练集文本和测试集文本数据转换成token单词编码。接着将转换成单词编码的训练集数据和测试集数据按样本分割成多个文件，一个文件代表一个样本。最后，我们可以根据文件名列表获取对应序号的样本内容，从而构建Dataset数据集。

```python
import numpy as np 
import pandas as pd 
from collections import OrderedDict
import re,string

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20 

train_data_path = 'data/imdb/train.tsv'
test_data_path = 'data/imdb/test.tsv'
train_token_path = 'data/imdb/train_token.tsv'
test_token_path =  'data/imdb/test_token.tsv'
train_samples_path = 'data/imdb/train_samples/'
test_samples_path =  'data/imdb/test_samples/'
```

首先我们构建词典，并保留最高频的MAX_WORDS个词。

```python
##构建词典

word_count_dict = {}

#清洗文本
def clean_text(text):
    lowercase = text.lower().replace("\n"," ")
    stripped_html = re.sub('<br />', ' ',lowercase)
    cleaned_punctuation = re.sub('[%s]'%re.escape(string.punctuation),'',stripped_html)
    return cleaned_punctuation

with open(train_data_path,"r",encoding = 'utf-8') as f:
    for line in f:
        label,text = line.split("\t")
        cleaned_text = clean_text(text)
        for word in cleaned_text.split(" "):
            word_count_dict[word] = word_count_dict.get(word,0)+1 

df_word_dict = pd.DataFrame(pd.Series(word_count_dict,name = "count"))
df_word_dict = df_word_dict.sort_values(by = "count",ascending =False)

df_word_dict = df_word_dict[0:MAX_WORDS-2] #  
df_word_dict["word_id"] = range(2,MAX_WORDS) #编号0和1分别留给未知词<unkown>和填充<padding>

word_id_dict = df_word_dict["word_id"].to_dict()

```

然后我们利用构建好的词典，将文本转换成token序号。

```python
#转换token

# 填充文本
def pad(data_list,pad_length):
    padded_list = data_list.copy()
    if len(data_list)> pad_length:
         padded_list = data_list[-pad_length:]
    if len(data_list)< pad_length:
         padded_list = [1]*(pad_length-len(data_list))+data_list
    return padded_list

def text_to_token(text_file,token_file):
    with open(text_file,"r",encoding = 'utf-8') as fin,\
      open(token_file,"w",encoding = 'utf-8') as fout:
        for line in fin:
            label,text = line.split("\t")
            cleaned_text = clean_text(text)
            word_token_list = [word_id_dict.get(word, 0) for word in cleaned_text.split(" ")]
            pad_list = pad(word_token_list,MAX_LEN)
            out_line = label+"\t"+" ".join([str(x) for x in pad_list])
            fout.write(out_line+"\n")
        
text_to_token(train_data_path,train_token_path)
text_to_token(test_data_path,test_token_path)

```

接着将token文本按照样本分割，每个文件存放一个样本的数据。

```python
# 分割样本
import os

if not os.path.exists(train_samples_path):
    os.mkdir(train_samples_path)
    
if not os.path.exists(test_samples_path):
    os.mkdir(test_samples_path)
    
    
def split_samples(token_path,samples_dir):
    with open(token_path,"r",encoding = 'utf-8') as fin:
        i = 0
        for line in fin:
            with open(samples_dir+"%d.txt"%i,"w",encoding = "utf-8") as fout:
                fout.write(line)
            i = i+1

split_samples(train_token_path,train_samples_path)
split_samples(test_token_path,test_samples_path)
```
一切准备就绪，我们可以创建数据集Dataset, 从文件名称列表中读取文件内容了。

```python
import os
class imdbDataset(Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)
    
    def __len__(self):
        return len(self.samples_paths)
    
    def __getitem__(self,index):
        path = self.samples_dir + self.samples_paths[index]
        with open(path,"r",encoding = "utf-8") as f:
            line = f.readline()
            label,tokens = line.split("\t")
            label = torch.tensor([float(label)],dtype = torch.float)
            feature = torch.tensor([int(x) for x in tokens.split(" ")],dtype = torch.long)
            return  (feature,label)
    
```

```python
ds_train = imdbDataset(train_samples_path)
ds_test = imdbDataset(test_samples_path)
```

```python
print(len(ds_train))
print(len(ds_test))
```

```python
dl_train = DataLoader(ds_train,batch_size = BATCH_SIZE,shuffle = True,num_workers=4)
dl_test = DataLoader(ds_test,batch_size = BATCH_SIZE,num_workers=4)

for features,labels in dl_train:
    print(features)
    print(labels)
    break
```

### 1-4: Dataloader
DataLoader能够控制batch的大小，batch中元素的采样方法，以及将batch结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据。

```python
DataLoader(
    dataset,
    batch_size=1,
    shuffle=False, 
    sampler=None, #样本采样函数，一般无需设置
    batch_sampler=None, #批次采样函数，一般无需设置。
    num_workers=0, #使用多进程读取数据，设置的进程数。
    collate_fn=None, #整理一个批次数据的函数。
    pin_memory=False, #是否设置为锁业内存。默认为False，锁业内存不会使用虚拟内存(硬盘)，从锁业内存拷贝到GPU上速度会更快。
    drop_last=False, #是否丢弃最后一个样本数量不足batch_size批次数据。
    timeout=0, #加载一个数据批次的最长等待时间，一般无需设置。
    worker_init_fn=None, #每个worker中dataset的初始化函数，常用于 IterableDataset。一般不使用。
    multiprocessing_context=None,
)
```

## 02: Build Model

