# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial - chapter 09 - petastorm-pyspark-pytorch

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. load parquet data into pytorch loader
# MAGIC
# MAGIC file path: `notebooks/images_data/silver/augmented`

# COMMAND ----------

# spark
from pyspark.sql.functions import lit
from pyspark.sql.types import BinaryType,StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions 
from pyspark.sql.types import *

#petastorm

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec 
    
    
import io
import os
import random
import numpy as np
from PIL import Image
from functools import partial 


# train images with pytorch
#from torchvision import transforms
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# import mlflow
import mlflow
import mlflow.pytorch





# COMMAND ----------

# start Spark session:

spark = SparkSession \
    .builder \
    .appName("Distributed Pytorch training") \
    .config("spark.memory.offHeap.enabled",True) \
    .config("spark.memory.offHeap.size","30g")\
    .getOrCreate()

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

# COMMAND ----------

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'petastorm_cache')

# COMMAND ----------

data_path = "images_data/silver/augmented"
mlflow_model_dir_path = "/"

# COMMAND ----------

# MAGIC %md
# MAGIC # Enable MLFlow tracking

# COMMAND ----------

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy

# COMMAND ----------

#Enable MLFlow tracking
mlflow.set_experiment(mlflow_model_dir_path)
# requires pytorch_lightning
mlflow.pytorch.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## params

# COMMAND ----------

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 1
#The number of **epochs** is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.
SAMPLE_SIZE = 5
NUM_EPOCHS = 1
NUM_EXECUTERS = 1
LEARNING_RATE=0.1
RANDOM_SEED_DEFAULT=0


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load preprocessed data

# COMMAND ----------

# Read the training data stored in parquet, limiting the dataset for the example
df_parquet = spark.read.parquet(data_path)
df = df_parquet.select(col("content"), col("label_index").cast(LongType())).limit(SAMPLE_SIZE)
  
num_classes = df.select("label_index").distinct().count()


# COMMAND ----------

# tmp since we are running localy on a sample
num_classes =4

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split to train and test

# COMMAND ----------

df_train , df_val = df.randomSplit([0.6,0.4], seed=12345)  

# COMMAND ----------

# convert the content vector into an array

# COMMAND ----------

df_train.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cache the Spark DataFrame using Petastorm Spark Converter

# COMMAND ----------

tmp_path = "file:/home/jovyan/petastorm_cache/"

# Set a cache directory on DBFS FUSE for intermediate data
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,tmp_path)


#train
converter_train = make_spark_converter(df_train)
#test
converter_val = make_spark_converter(df_val)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Petastorm prepreocess
# MAGIC used during materlizing spark dataframe with petastorm and bridging to TensorFlow

# COMMAND ----------

import torchvision, torch
from torchvision import datasets, models, transforms

def preprocess(image):
  """
  Preprocess an image file bytes for MobileNetV2 (ImageNet) - using torchvision transform and normalize.
  """
#   transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
    
  image = Image.open(io.BytesIO(image)).resize([224, 224])
  arr = np.array(image)
#   image_array_byte = np.array(image)
  # convert byte array to float array for PyTorch backward computation
#   image_array_float = [float(a) for a in image_array_byte]
  #image_array = keras.preprocessing.image.img_to_array(image)
#   pytorch_normalize =  transform(image)

  return arr



def transform_row(pd_batch):
  """
  The input and output of this function are pandas dataframes.
  """
  pd_batch['content'] = pd_batch['content'].map(lambda x: preprocess(x))
    

  return pd_batch

# The output shape of the `TransformSpec` is not automatically known by petastorm, 
# so you need to specify the shape for new columns in `edit_fields` and specify the order of 
# the output columns in `selected_fields`.
transform_spec_fn = TransformSpec(
  func=transform_row, 
  selected_fields=['content', 'label_index']
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Get the model MobileNetV2
# MAGIC #### Get the model MobileNetV2 from torch hub
# MAGIC and only retraining its final layer to fit your needs.

# COMMAND ----------

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Set PyTorch environment for distributed training

# COMMAND ----------

# as described in the book - always set the seed for the machine in distributed training
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# COMMAND ----------

set_random_seeds(random_seed=RANDOM_SEED_DEFAULT)

# COMMAND ----------

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


# COMMAND ----------

# Initializes the distributed backend which will take care of sychronizing nodes/GPUs

# choose the one relevant to your backend.

# torch.distributed.init_process_group(backend="nccl")
# torch.distributed.init_process_group(backend="gloo")

# COMMAND ----------

# if next(model.parameters()).is_cuda:
#    device = torch.device("cuda:{}".format(local_rank))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. set the train function

# COMMAND ----------

def train(data_loader, steps=100, lr=0.0005, momentum=0.5):
    model.train()
    
    
    ##  train the model 
    for data in data_loader:
        print(data)
    return model

# COMMAND ----------

def test(model, test_loader,device):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy
    return accuracy

# COMMAND ----------

setup(1,1)
machine_rank = dist.get_rank()

# COMMAND ----------

model = model.to(1)

# COMMAND ----------

# combine PyTorch distributed data parallel with Petastorm
# this approach is replicating the model over the training machines
# where each machine operates on one chunk of the data, as described in Chatper 9.
    

setup(rank=0, world_size=1)    
model = model.to(machine_rank)
ddp_model = DDP(model, device_ids=[machine_rank])
    
def train_and_evaluate(_=None):
    with converter_train.make_torch_dataloader(transform_spec=transform_spec_fn ,batch_size=BATCH_SIZE) as loader:
            outputs = ddp_model(loader)
    
    with converter_val.make_torch_dataloader(transform_spec=transform_spec_fn,batch_size=BATCH_SIZE,num_epochs=1) as loader:
            accuracy = test(model, loader)
            return accuracy

# COMMAND ----------

 accuracy = train_and_evaluate()

# COMMAND ----------

accuracy

# COMMAND ----------

להמשיך מכאן - נראה שצריך לסדר את העניין של המערכת עצמה בשביל שזה יהיה מוכן לטובת הרצה בצורה distributed