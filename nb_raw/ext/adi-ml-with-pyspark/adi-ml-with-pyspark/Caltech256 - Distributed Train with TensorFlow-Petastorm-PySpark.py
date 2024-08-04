# Databricks notebook source
# MAGIC %md
# MAGIC # Caltech 256 - Distributed Train with TensorFlow-Petastorm-PySpark

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Init environment
# MAGIC 2. Load sample data
# MAGIC 3. Split to train and test
# MAGIC 4. Convert data to TensorFlow Dataset
# MAGIC 5. Define function to Feed the data into a distributed TensorFlow model for training
# MAGIC 6. Run the experiment with MLFlow and HVD
# MAGIC 7. Inference, serve the trained model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.Init environment
# MAGIC Ingest the data in batch from storage

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions 
from pyspark.sql.types import *


from petastorm.spark import SparkDatasetConverter, make_spark_converter
 
import io
import numpy as np
from PIL import Image
from functools import partial 
from petastorm import TransformSpec
#from torchvision import transforms
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import tensorflow as tf


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


import mlflow
import mlflow.keras
import mlflow.tensorflow



# COMMAND ----------

# start Spark session:
#     .config("spark.jars.packages", "org.mlflow.mlflow-spark") \

spark = SparkSession \
    .builder \
    .appName("Distributed training with TensorFlow") \
    .getOrCreate()

# COMMAND ----------

data_path = "images_data/silver/augmented"
mlflow_model_dir_path = "/"

# COMMAND ----------

# MAGIC %md
# MAGIC ###Enable mlflow tracking

# COMMAND ----------

#Enable MLFlow tracking
mlflow.set_experiment(mlflow_model_dir_path)


mlflow.tensorflow.autolog()
# mlflow.spark.autolog()



# COMMAND ----------

# MAGIC %md
# MAGIC ### params

# COMMAND ----------

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 1
#The number of **epochs** is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.
SAMPLE_SIZE = 5
NUM_EPOCHS = 1
NUM_EXECUTERS = 1


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load preprocessed data

# COMMAND ----------

# Read the training data stored in parquet, limiting the dataset for the example
df_parquet = spark.read.parquet(data_path)  

# COMMAND ----------

df_parquet.printSchema()

# COMMAND ----------

df_parquet.select("label_index","label").distinct().show()

# COMMAND ----------

df = df_parquet.select(col("content"), col("label_index").cast(LongType())).limit(SAMPLE_SIZE)
num_classes = df.select("label_index").distinct().count()

# COMMAND ----------

num_classes =4

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split to train and test

# COMMAND ----------

df_train , df_val = df.randomSplit([0.6,0.4], seed=12345)  

# COMMAND ----------

df_train.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cache the Spark DataFrame using Petastorm Spark Converter

# COMMAND ----------

tmp_path = "file:/home/jovyan/petastorm_cache/"

# Set a cache directory on DBFS FUSE for intermediate data
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,tmp_path)

# TIP: Use a low value for parquet_row_group_bytes. The detafault of 32 MiB can be too high for larger datasets. Using 1MB instead.
#train
converter_train = make_spark_converter(df_train, parquet_row_group_size_bytes=32000000)
#test
converter_val = make_spark_converter(df_val, parquet_row_group_size_bytes=32000000)


# COMMAND ----------

# MAGIC %md
# MAGIC tensorflow batch size needs to be smaller than training data size

# COMMAND ----------

print(f"train: {len(converter_train)}, test: {len(converter_val)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Get the model MobileNetV2
# MAGIC #### Get the model MobileNetV2 from tensorflow.keras
# MAGIC and only retraint it's final layer to fit our needs.

# COMMAND ----------

# First, load the model and inspect the structure of the model.
MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet').summary()

# COMMAND ----------

def get_model(lr=0.001):

  # Create the base model from the pre-trained model MobileNet V2
  base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
  # Freeze parameters in the feature extraction layers
  base_model.trainable = False
  
  # Add a new classifier layer for transfer learning
  global_average_layer = keras.layers.GlobalAveragePooling2D()
  prediction_layer = keras.layers.Dense(num_classes)
  
  model = keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
  ])
  return model

# COMMAND ----------

df_train.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train TensorFlow with sample data on single-machine
# MAGIC Often, during the development, a data scientist will move from training the model on on single machine to training it on multiple machines on a larger data. 
# MAGIC But first, to understand how to connect the dots and the impact of various algorithms, we will start with training on a single machine

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feed the data into a distributed TensorFlow model for training
# MAGIC
# MAGIC Use HorovodRunner for distributed training.
# MAGIC
# MAGIC Use the default value of parameter `num_epochs=None` to generate infinite batches of data to avoid handling the last incomplete batch. This is particularly useful in the distributed training scenario, where you need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Develop the pyfunc wrapper for the model

# COMMAND ----------

# TIP: Create custom Python pyfunc model that transforms and predicts on inference data
# Allows the inference pipeline to be independent of the model framework used in training pipeline
class KerasCNNModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model_path):
    self.model_path = model_path

  def load_context(self, context):
    # Load the Keras-native representation of the MLflow
    # model
    print(self.model_path)
    self.model = mlflow.keras.load_model(
        model_uri=self.model_path)

  def predict(self, context, model_input):
    import tensorflow as tf
    import json

    class_def = {
      0: '212.teapot', 
      1: '234.tweezer', 
      2: '196.spaghetti', 
      3: '249.yo-yo', 
    }

    model_input['origin'] = model_input['origin'].str.replace("dbfs:","/dbfs")
    images = model_input['origin']

    rtn_df = model_input.iloc[:,0:1]
    rtn_df['prediction'] = None
    rtn_df['probabilities'] = None

    for index, row in model_input.iterrows():
      image = np.round(np.array(Image.open(row['origin']).resize((224,224)),dtype=np.float32))
      img = tf.reshape(image, shape=[-1, 224, 224, 3])
      class_probs = self.model.predict(img)
      classes = np.argmax(class_probs, axis=1)
      class_prob_dict = dict()
      for key, val in class_def.items():
        class_prob_dict[val] = np.round(np.float(class_probs[0][int(key)]), 3).tolist()
      rtn_df.loc[index,'prediction'] = classes[0]
      rtn_df.loc[index,'probabilities'] = json.dumps(class_prob_dict)

    return rtn_df[['prediction', 'probabilities']].values.tolist()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Petastorm prepreocess
# MAGIC used during materlizing spark dataframe with petastorm and bridging to TensorFlow

# COMMAND ----------

def preprocess(grayscale_image):
  """
  Preprocess an image file bytes for MobileNetV2 (ImageNet).
  """
  image = Image.open(io.BytesIO(grayscale_image)).resize([224, 224])
  image_array = keras.preprocessing.image.img_to_array(image)
  return preprocess_input(image_array)

def transform_row(pd_batch):
  """
  The input and output of this function are pandas dataframes.
  """
  pd_batch['features'] = pd_batch['content'].map(lambda x: preprocess(x))
  pd_batch = pd_batch.drop(labels=['content'], axis=1)
  return pd_batch

# The output shape of the `TransformSpec` is not automatically known by petastorm, 
# so you need to specify the shape for new columns in `edit_fields` and specify the order of 
# the output columns in `selected_fields`.
transform_spec_fn = TransformSpec(
  func=transform_row, 
  edit_fields=[('features', np.uint8 , IMG_SHAPE, False)], 
  selected_fields=['features', 'label_index']
)

# COMMAND ----------

# MAGIC %md
# MAGIC Spark converter returns a TFDatasetContextManager or a TorchDatasetContextManager for PyTorch depends on make function `make_tf_dataset`
# MAGIC
# MAGIC https://github.com/uber/petastorm/blob/master/petastorm/spark/spark_dataset_converter.py

# COMMAND ----------

# MAGIC %md
# MAGIC Image properties include number of rows, columns, and channels; type of image data; number of pixels; etc.
# MAGIC
# MAGIC The **shape** of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is in color/RGB):
# MAGIC
# MAGIC
# MAGIC The number of **epochs** is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train a dist model

# COMMAND ----------


# configure the strategy accoring to the existing physical devices available
if tf.config.list_physical_devices('GPU'):
  strategy = tf.distribute.MirroredStrategy()
else:  # Use the Default Strategy
  strategy = tf.distribute.get_strategy()


model = get_model(lr=0.001)
model.compile(optimizer="SGD", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       batch_size=BATCH_SIZE) as train_dataset, \
       converter_val.make_tf_dataset(transform_spec=transform_spec_fn,
                                    batch_size=BATCH_SIZE) as val_dataset:
    
    # everything from here - runs within TF strategy scope
    with strategy.scope():
        # tf.keras only accept tuples, not namedtuples
        train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
        steps_per_epoch = len(converter_train) // (BATCH_SIZE)

        val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
        validation_steps = max(1, len(converter_val) // (BATCH_SIZE ))

        hist = model.fit(train_dataset, 
                         steps_per_epoch=steps_per_epoch,
                         epochs=NUM_EPOCHS,
                         validation_data=val_dataset,
                         validation_steps=validation_steps,
                         verbose=2)

# COMMAND ----------

# MAGIC %md
# MAGIC Hist of of type -tensorflow.python.keras.callbacks.History : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History?hl=iw

# COMMAND ----------

hist.history.keys()
hist.params
hist.model

# COMMAND ----------

new_model = hist.model 