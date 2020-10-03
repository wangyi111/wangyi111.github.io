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


```python
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


```python
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


```python
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


```python
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


```python
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


```python
ds5 = tf.data.TextLineDataset(
    filenames = ["../data/titanic/train.csv","../data/titanic/test.csv"]
    ).skip(1) # Omitting the header on the first line

for line in ds5.take(5):
    print(line)
```

### 1-6: file path


```python
ds6 = tf.data.Dataset.list_files("../data/cifar2/train/*/*.jpg")
for file in ds6.take(5):
    print(file)
```


```python
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


```python
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


```python
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


```python
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


```python
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


```python
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


```python
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


```python
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


```python
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


```python
model = create_model() # create model using methods listed above
model.compile(optimizer=optimizers.Nadam(),loss=losses.SparseCategoricalCrossentropy(),metrics=['accuracy']) # compile model

logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1) # tensorboard

history = model.fit(ds_train,validation_data = ds_test,epochs = 10,callbacks=[tensorboard_callback]) # train model
```

### 3-2: train_on_batch

This pre-defined method allows fine-controlling to the training procedure for each batch without the callbacks, which is even more flexible than fit method.


```python
model = create_model() # create model using methods listed above
model.compile(optimizer=optimizers.Nadam(),loss=losses.SparseCategoricalCrossentropy(),metrics=['accuracy']) # compile model
```


```python
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


```python
train_model(model,ds_train,ds_test,10)
```

### 3-3: customized training loop

Re-compilation of the model is not required in the customized training loop, just back-propagate the iterative parameters through the optimizer according to the loss function, which gives us the highest flexibility.


```python
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


```python
train_model(model,ds_train,ds_test,10)
```

### 3-4: Use GPU
Tensorflow uses GPU by default, but to control the usage of GPUs we need to add some codes.

1, Set GPU and memory.
```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    # 或者也可以设置GPU显存为固定使用量(例如：4G)
    #tf.config.experimental.set_virtual_device_configuration(gpu0,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
    tf.config.set_visible_devices([gpu0],"GPU") 
```
2, Use multiple GPUs with MirroredStrategy().

MirroredStrategy过程简介：
* 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
* 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
* N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
* 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
* 使用梯度求和的结果更新本地变量（镜像变量）；
* 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。
```python
#增加以下两行代码
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
    model = create_model()
    model.summary()
    model = compile_model(model)
    
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)  
```
3, Use colab TPU.
```python
#增加以下6行代码
import os
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    model.summary()
    model = compile_model(model)
```

## 04: Save and use model

### 4-1: Save the whole model

Save the whole model, including the model's architecture, weight values, training config, optimizer and its states (useful for web/mobile usage).

```python
# save model as h5 file
model.save('mymodel.h5')

# load model
new_model = tf.keras.models.load_model('mymodel.h5')
```


```python
# save model as SavedModel (recommended)
model.save('tf_model_savedmodel', save_format="tf")

# load model
new_model = tf.keras.models.load_model('tf_model_savedmodel')
```

### 4-2: Save model's architecture and weights

```python
# save model's architecture
json_config = model.to_json()
with open('model_config.json','w') as json_file:
  json_file.write(json_config)

# save model's weights
#model.save_weights('model_weights.h5')
#model.save_weights('./checkpoints/my_checkpoint')
model.save_weights('model_weights',save_format='tf')

# load model
with open('model_config.json') as json_file:
  json_config = json_file.read()
new_model = tf.keras.models.model_from_json(json_config)
#new_model.load_weights('model_weights.h5')
#new_model.load_weights('./checkpoints/my_checkpoint')
new_model.load_weights('model_weights')
```

Note: If the training process is customized (i.e. not using ```model.fit```), we should set model's input shape before training.

```python
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

### 4-3: Save weights while training

You can use a trained model without having to retrain it, or pick-up training where you left off in case the training process was interrupted. The `tf.keras.callbacks.ModelCheckpoint` callback allows you to continually save the model both during and at the end of training.

Create a `tf.keras.callbacks.ModelCheckpoint` callback that saves weights only during training:

```python
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training
# This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch.

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.
```

Then load saved weights and evaluate on a new model:
```python
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

The callback provides several options to provide unique names for checkpoints and adjust the checkpointing frequency.

Train a new model, and save uniquely named checkpoints once every five epochs:
```python
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)
```
Now load the latest checkpoint to test:
```python
# Create a new model instance
model = create_model()
# Load the previously saved weights
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
# Re-evaluate the model
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

The above code stores the weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format. Checkpoints contain:
* One or more shards that contain your model's weights.
* An index file that indicates which weights are stored in which shard.
If you are training a model on a single machine, you'll have one shard with the suffix: `.data-00000-of-00001`.