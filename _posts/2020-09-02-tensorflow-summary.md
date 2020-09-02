---
title: 'Tensorflow: Brief Summary'
permalink: /posts/2020/09/tensorflow-summary/
categories: programming
tags:
  - tensorflow
toc: true
toc_label: "CONTENT"
---

This is a brief summary for the implementation of tensorflow, from data preparation to the usage of trained model.

## 01: Prepare Data

In tensorflow the API tf.data constructs input data pipeline to help manage huge volume of data with various formats and conversions.

Data pipeline could be constructed through following methods: numpy array, pandas DataFrame, Python generator, csv file, text file, file path, tfrecords file.

Among these methods, the most popular ones are: numpy array, pandas DataFrame and file path. For very large data volume, the tfrecord is useful. The advantage of using tfrecords files is its small volume after compression, its convenient sharing through the Internet, and the fast speed of loading.

### 1-1: Numpy Array


```
import tensorflow as tf
import numpy as np 
from sklearn import datasets 
iris = datasets.load_iris()

ds1 = tf.data.Dataset.from_tensor_slices((iris["data"],iris["target"])) # dataset from numpy array
for features,label in ds1.take(5):
    print(features,label)
```

    tf.Tensor([5.1 3.5 1.4 0.2], shape=(4,), dtype=float64) tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor([4.9 3.  1.4 0.2], shape=(4,), dtype=float64) tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor([4.7 3.2 1.3 0.2], shape=(4,), dtype=float64) tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor([4.6 3.1 1.5 0.2], shape=(4,), dtype=float64) tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor([5.  3.6 1.4 0.2], shape=(4,), dtype=float64) tf.Tensor(0, shape=(), dtype=int64)
    

### 1-2: Pandas dataframe


```
from sklearn import datasets 
import pandas as pd

iris = datasets.load_iris()
dfiris = pd.DataFrame(iris["data"],columns = iris.feature_names)
ds2 = tf.data.Dataset.from_tensor_slices((dfiris.to_dict("list"),iris["target"]))

for features,label in ds2.take(3):
    print(features,label)
```

    {'sepal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=5.1>, 'sepal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=3.5>, 'petal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=1.4>, 'petal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=0.2>} tf.Tensor(0, shape=(), dtype=int64)
    {'sepal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=4.9>, 'sepal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=3.0>, 'petal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=1.4>, 'petal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=0.2>} tf.Tensor(0, shape=(), dtype=int64)
    {'sepal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=4.7>, 'sepal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=3.2>, 'petal length (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=1.3>, 'petal width (cm)': <tf.Tensor: shape=(), dtype=float32, numpy=0.2>} tf.Tensor(0, shape=(), dtype=int64)
    

### 1-3: Python generator


```
import tensorflow as tf
from matplotlib import pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Defining a generator to read image from a folder
image_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
                    "../data/cifar2/test/",
                    target_size=(32, 32),
                    batch_size=20,
                    class_mode='binary')

classdict = image_generator.class_indices
print(classdict)

def generator():
    for features,label in image_generator:
        yield (features,label)

ds3 = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.int32))
```


```
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.figure(figsize=(6,6)) 
for i,(img,label) in enumerate(ds3.unbatch().take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()
```

### 1-4: csv


```
ds4 = tf.data.experimental.make_csv_dataset(
      file_pattern = ["../data/titanic/train.csv","../data/titanic/test.csv"],
      batch_size=3, 
      label_name="Survived",
      na_value="",
      num_epochs=1,
      ignore_errors=True)

for data,label in ds4.take(2):
    print(data,label)
```

### 1-5: txt


```
ds5 = tf.data.TextLineDataset(
    filenames = ["../data/titanic/train.csv","../data/titanic/test.csv"]
    ).skip(1) # Omitting the header on the first line

for line in ds5.take(5):
    print(line)
```

### 1-6: file path


```
ds6 = tf.data.Dataset.list_files("../data/cifar2/train/*/*.jpg")
for file in ds6.take(5):
    print(file)
```


```
from matplotlib import pyplot as plt 
def load_image(img_path,size = (32,32)):
    label = 1 if tf.strings.regex_full_match(img_path,".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img) # Note that we are using jpeg format
    img = tf.image.resize(img,size)
    return(img,label)

%matplotlib inline
%config InlineBackend.figure_format = 'svg'
for i,(img,label) in enumerate(ds6.map(load_image).take(2)):
    plt.figure(i)
    plt.imshow((img/255.0).numpy())
    plt.title("label = %d"%label)
    plt.xticks([])
    plt.yticks([])
```

### 1-7: tfrecords

For the data pipeline with tfrecords file it is a little bit complicated, as we have to first create tfrecords file, and then read it to tf.data.dataset.


```
import os
import numpy as np

# inpath is the original data path; outpath: output path of the TFRecord file
def create_tfrecords(inpath,outpath): 
    writer = tf.io.TFRecordWriter(outpath)
    dirs = os.listdir(inpath)
    for index, name in enumerate(dirs):
        class_path = inpath +"/"+ name+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            #img = tf.image.decode_image(img)
            #img = tf.image.encode_jpeg(img) # Use jpeg format for all the compressions
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
               }))
            writer.write(example.SerializeToString())
    writer.close()
    
create_tfrecords("../data/cifar2/test/","../data/cifar2_test.tfrecords/")
```


```
from matplotlib import pyplot as plt 

def parse_example(proto):
    description ={ 'img_raw' : tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)} 
    example = tf.io.parse_single_example(proto, description)
    img = tf.image.decode_jpeg(example["img_raw"])   # Note that we are using jpeg format
    img = tf.image.resize(img, (32,32))
    label = example["label"]
    return(img,label)

ds7 = tf.data.TFRecordDataset("../data/cifar2_test.tfrecords").map(parse_example).shuffle(3000)

```


```
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.figure(figsize=(6,6)) 
for i,(img,label) in enumerate(ds7.take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow((img/255.0).numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()
```

Meanwhile, tf.data.dataset also has many functions for data conversion/preprocessing. Examples are listed as follows:

* `map`: projecting the conversion function to every element in the dataset.

* `flat_map`: projecting the conversion function to every element in the dataset, and flatten the embedded Dataset.

* `interleave`: similar as `flat_map` but interleaves the data from different sources.

* `filter`: filter certain elements.

* `zip`: zipping two Datasets with the same length.

* `concatenate`: concatenating two Datasets.

* `reduce`: executing operation of reducing.

* `batch`: constructing batches and release one batch each time; there will be one more rank comparing to the original data; the inverse operation is `unbatch`.

* `padded_batch`: constructing batches, similar as `batch`, but can achieve padded shape.

* `window`: constructing sliding window, and return Dataset of Dataset.

* `shuffle`: shuffling the order of the data.

* `repeat`: repeat the data certain times; if no argument is specified, repeat data with infinitive times.

* `shard`: sampling the elements starting from a certain position with fixed distance.

* `take`: sampling the first few elements from a certain position.

## 02: Build model

There are three ways of modeling: using `Sequential` to construct model with the order of layers, using functional APIs to construct model with arbitrary structure, using child class inheriting from the base class `Model`.

For the models with sequenced structure, `Sequential` method should be given the highest priority.

For the models with nonsequenced structures such as multiple input/output, shared weights, or residual connections, modeling with functional API is recommended.

Modeling through child class of `Model` should be AVOIDED unless with special requirements. This method is flexible, but also fallible.

### 2-1: Sequential


```
tf.keras.backend.clear_session()

model = models.Sequential()

model.add(layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN))
model.add(layers.Conv1D(filters = 64,kernel_size = 5,activation = "relu"))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = "relu"))
model.add(layers.MaxPool1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation = "sigmoid"))

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

model.summary()
```

### 2-2: functional API


```
tf.keras.backend.clear_session()

inputs = layers.Input(shape=[MAX_LEN])
x  = layers.Embedding(MAX_WORDS,7)(inputs)

branch1 = layers.SeparableConv1D(64,3,activation="relu")(x)
branch1 = layers.MaxPool1D(3)(branch1)
branch1 = layers.SeparableConv1D(32,3,activation="relu")(branch1)
branch1 = layers.GlobalMaxPool1D()(branch1)

branch2 = layers.SeparableConv1D(64,5,activation="relu")(x)
branch2 = layers.MaxPool1D(5)(branch2)
branch2 = layers.SeparableConv1D(32,5,activation="relu")(branch2)
branch2 = layers.GlobalMaxPool1D()(branch2)

branch3 = layers.SeparableConv1D(64,7,activation="relu")(x)
branch3 = layers.MaxPool1D(7)(branch3)
branch3 = layers.SeparableConv1D(32,7,activation="relu")(branch3)
branch3 = layers.GlobalMaxPool1D()(branch3)

concat = layers.Concatenate()([branch1,branch2,branch3])
outputs = layers.Dense(1,activation = "sigmoid")(concat)

model = models.Model(inputs = inputs,outputs = outputs)

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

model.summary()
```

### 2-3: Customized Modeling Using Child Class of Model


```
# Define a customized residual module as Layer

class ResBlock(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self,input_shape):
        self.conv1 = layers.Conv1D(filters=64,kernel_size=self.kernel_size,
                                   activation = "relu",padding="same")
        self.conv2 = layers.Conv1D(filters=32,kernel_size=self.kernel_size,
                                   activation = "relu",padding="same")
        self.conv3 = layers.Conv1D(filters=input_shape[-1],
                                   kernel_size=self.kernel_size,activation = "relu",padding="same")
        self.maxpool = layers.MaxPool1D(2)
        super(ResBlock,self).build(input_shape) # Identical to self.built = True
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.Add()([inputs,x])
        x = self.maxpool(x)
        return x
    
    # Need to define get_config method in order to sequentialize the model constructed from the customized Layer by Functional API.
    def get_config(self):  
        config = super(ResBlock, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
```


```
# Customized model, which could also be implemented by Sequential or Functional API

class ImdbModel(models.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()
        
    def build(self,input_shape):
        self.embedding = layers.Embedding(MAX_WORDS,7)
        self.block1 = ResBlock(7)
        self.block2 = ResBlock(5)
        self.dense = layers.Dense(1,activation = "sigmoid")
        super(ImdbModel,self).build(input_shape)
    
    def call(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = layers.Flatten()(x)
        x = self.dense(x)
        return(x)
```


```
tf.keras.backend.clear_session()

model = ImdbModel()
model.build(input_shape =(None,200))
model.summary()

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])
```

## 03: Train model

There are three ways of model training: using pre-defined fit method, using pre-defined tran_on_batch method, using customized training loop.

Note: fit_generator method is not recommended in tf.keras since it has been merged into fit.

### 3-1: model.fit

build model -> compile model -> train model


```
model = create_model() # create model using methods listed above
model.compile(optimizer=optimizers.Nadam(),loss=losses.SparseCategoricalCrossentropy(),metrics=['accuracy']) # compile model

logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1) # tensorboard

history = model.fit(ds_train,validation_data = ds_test,epochs = 10,callbacks=[tensorboard_callback]) # train model
```

### 3-2: train_on_batch

This pre-defined method allows fine-controlling to the training procedure for each batch without the callbacks, which is even more flexible than fit method.


```
model = create_model() # create model using methods listed above
model.compile(optimizer=optimizers.Nadam(),loss=losses.SparseCategoricalCrossentropy(),metrics=['accuracy']) # compile model
```


```
def train_model(model,ds_train,ds_valid,epoches):

    for epoch in tf.range(1,epoches+1):
        model.reset_metrics()
        
        # Reduce learning rate at the late stage of training.
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr/2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")
        
        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y,reset_metrics=False)
            
        if epoch%1 ==0:
            printbar()
            tf.print("epoch = ",epoch)
            print("train:",dict(zip(model.metrics_names,train_result)))
            print("valid:",dict(zip(model.metrics_names,valid_result)))
            print("")
```


```
train_model(model,ds_train,ds_test,10)
```

### 3-3: customized training loop

Re-compilation of the model is not required in the customized training loop, just back-propagate the iterative parameters through the optimizer according to the loss function, which gives us the highest flexibility.


```
optimizer = optimizers.Nadam()
loss_func = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features,training = True) # prediction
        loss = loss_func(labels, predictions) # calculate loss
    gradients = tape.gradient(loss, model.trainable_variables) # gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # gradient descent -> optimizer

    train_loss.update_state(loss) # update loss
    train_metric.update_state(labels, predictions) # update metric
    

@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)
    

def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        
        for features, labels in ds_train:
            train_step(model,features,labels)

        for features, labels in ds_valid:
            valid_step(model,features,labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        
        if epoch%1 ==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")
            
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

```


```
train_model(model,ds_train,ds_test,10)
```

## 04: Save and use model

4-1. Save the whole model, including the model's architecture, weight values, training config, optimizer and its states.




```
# save model as h5 file
model.save('mymodel.h5')

# load model
new_model = tf.keras.models.load_model('mymodel.h5')
```


```
# save model as SavedModel (recommended)
model.save('tf_model_savedmodel', save_format="tf")

# load model
new_model = tf.keras.models.load_model('tf_model_savedmodel')
```

4-2. Save model's architecture and weights.


```
# save model's architecture
json_config = model.to_json()
with open('model_config.json','w') as json_file:
  json_file.write(json_config)
# save model's weights
#model.save_weights('model_weights.h5')
model.save_weights('model_weights',save_format='tf')
# load model
with open('model_config.json') as json_file:
  json_config = json_file.read()
new_model = tf.keras.models.model_from_json(json_config)
#new_model.load_weights('model_weights.h5')
new_model.load_weights('model_weights')
```

Note: If the training process is customized (i.e. not using ```model.fit```), we should set model's input shape before training.


```
# create model
model = KeypointNetwork()  

# set input shape
shape = tf.TensorSpec(shape = (batch_size,210), dtype=tf.dtypes.float32, name=None)
model._set_inputs(shape)  
 
# training
for epoch in range(1,epochs+1):
    ... ...
    ... ...
 
 
# save model
model.save('./model-weights/checkpoint_weights',save_format='tf')
```
