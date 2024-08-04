# Databricks notebook source
# MAGIC %md
# MAGIC simple example of how to create a petstorm store, as discussed in chapter 7

# COMMAND ----------

SAMPLE_SIZE = 1

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

from pyspark.sql.functions import col
from pyspark.sql.types import *

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec 

import tensorflow as tf

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

data_path = "images_data/silver/augmented"
# Read the data stored in parquet, limiting the dataset for the example
df_parquet = spark.read.parquet(data_path)
df = df_parquet.select(col("content"), col("label_index").cast(LongType())).limit(SAMPLE_SIZE)


# COMMAND ----------

df.printSchema()

# COMMAND ----------

expected_df = df.collect()

# COMMAND ----------

expected_df[0][1]

# COMMAND ----------

import tensorflow as tf


converter = make_spark_converter(df)
with converter.make_tf_dataset(transform_spec=transform_spec_fn) as dataset:
        iterator = iter(dataset)
        print(iterator.element_spec)
        
        tensor = iterator.get_next()
        print(tensor)

# COMMAND ----------

# The schema defines how the dataset schema looks like
imageSchema = Unischema('caltech256schema', [
    UnischemaField('content', np.uint8, (224, 224, 3) ,False),
    UnischemaField('label_index', np.int32, (), ScalarCodec(LongType()), False)])

# COMMAND ----------

imageSchema

# COMMAND ----------

rows_rdd =  df.rdd.map(lambda x: dict_to_spark_row(imageSchema, x))

rows_rdd.take(1)

# COMMAND ----------

sc = spark.sparkContext
output_url = "images_data/petastorm"
rowgroup_size_mb = 256

    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
rows_count = 10

with materialize_dataset(spark, output_url, imageSchema, rowgroup_size_mb):
        spark.createDataFrame(rows_rdd, imageSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Barrier code example

# COMMAND ----------

from pyspark import BarrierTaskContext


def stage_logic(row):
    context = BarrierTaskContext.get()
    # some logic that needs to be coordinated
    context.barrier()
    return row

# COMMAND ----------

barrierRdd = df.rdd.barrier()
rdd = barrierRdd.mapPartitions(lambda x: stage_logic( x))

# COMMAND ----------

# MAGIC %md
# MAGIC ## to delete! 

# COMMAND ----------

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

# The schema defines how the dataset schema looks like
HelloWorldSchema = Unischema('HelloWorldSchema', [
    UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('image1', np.uint8, (128, 256, 3), CompressedImageCodec('png'), False),
    UnischemaField('array_4d', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
])


def row_generator(x):
    """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
    return {'id': x,
            'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3)),
            'array_4d': np.random.randint(0, 255, dtype=np.uint8, size=(4, 128, 30, 3))}


def generate_petastorm_dataset(output_url='file://hello_world_dataset'):
    rowgroup_size_mb = 256

    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark.sparkContext

    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    rows_count = 10
    with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):

        rows_rdd = sc.parallelize(range(rows_count))\
            .map(row_generator)\
            .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

        spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)
        
generate_petastorm_dataset()

# COMMAND ----------

    # Cleanup
    converter_train.delete()
    converter_test.delete()
    spark.stop()