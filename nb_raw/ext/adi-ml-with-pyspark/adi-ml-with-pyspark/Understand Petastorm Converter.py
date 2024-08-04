# Databricks notebook source
# MAGIC %md
# MAGIC # Understanding Petastorm converter and types ( Chapter 7 )

# COMMAND ----------

SAMPLE_SIZE = 50

# COMMAND ----------

# MAGIC %md
# MAGIC Define schema

# COMMAND ----------

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec 

from pyspark.sql.functions import col
from pyspark.sql.types import *


from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType,
                               DoubleType, FloatType, IntegerType, LongType,
                               ShortType, StringType, StructField, StructType)


# COMMAND ----------

# start Spark session:

spark = SparkSession \
    .builder \
    .appName("Create petastorm store") \
    .config("spark.memory.offHeap.enabled",True) \
    .config("spark.memory.offHeap.size","30g")\
    .getOrCreate()

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file:/home/jovyan/petastorm_tmp_cache')

# COMMAND ----------

# MAGIC %md
# MAGIC ## understand primitives

# COMMAND ----------

 schema = StructType([
        StructField("bool_col", BooleanType(), False),
        StructField("float_col", FloatType(), False),
        StructField("double_col", DoubleType(), False),
        StructField("short_col", ShortType(), False),
        StructField("int_col", IntegerType(), False),
        StructField("long_col", LongType(), False),
        StructField("str_col", StringType(), False),
        StructField("bin_col", BinaryType(), False),
        StructField("byte_col", ByteType(), False),
    ])

# COMMAND ----------

df = spark.createDataFrame(
        [(True, 0.12, 432.1, 5, 5, 0, "hello",
          bytearray(b"spark\x01\x02"), -128),
         
         (False, 123.45, 0.987, 9, 908, 765, "petastorm",
          bytearray(b"\x0012345"), 127)],
        schema=schema).coalesce(1)

    # If we use numPartition > 1 in coalesce, the order of the loaded dataset would
    # be non-deterministic.
# just for the learning phase - DO NOT USE IN PRODUCTION!    
expected_df = df.collect()

# COMMAND ----------

expected_df

# COMMAND ----------

# MAGIC %md
# MAGIC  ## Test TensorFlow dataset

# COMMAND ----------

# create tf operations graph

# COMMAND ----------

import tensorflow as tf


converter = make_spark_converter(df)
with converter.make_tf_dataset() as dataset:
        iterator = iter(dataset)
        print(iterator.element_spec)
        
        tensor = iterator.get_next()
        print(tensor)

# COMMAND ----------

