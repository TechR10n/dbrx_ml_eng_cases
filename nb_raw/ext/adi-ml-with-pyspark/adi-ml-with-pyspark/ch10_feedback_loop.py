# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 10 - the feedback loop
# MAGIC
# MAGIC As you learned in chapter 10 about the feedback loop, let's look at an example that shows how it works.
# MAGIC
# MAGIC Our system uses Apache Spark for batch and stream processing. Our machine learning model was built with Spark.
# MAGIC Spark is serving the machine learning model as part of the work with stractured streaming.
# MAGIC
# MAGIC For the example. we will generate some data.

# COMMAND ----------



# machine learning model pipeline

# stream data spark

# table with actual and prediction

# fake alert


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import expr

spark = SparkSession \
    .builder \
    .appName("machine_learning_streaming") \
    .getOrCreate()

# COMMAND ----------

source_stream = spark.readStream.format("csv").option("header",True)\
    .schema(schema).option("ignoreLeadingWhiteSpace",True)\
    .option("mode","dropMalformed").option("maxFilesPerTrigger",1)\
    .load({data_path}).withColumnRenamed("output","label")


some_model = ...

# COMMAND ----------

stream_predict = some_model.transform(source_stream).select('probability','prediction')

# COMMAND ----------

stream_actual = other_source_stream_with_actual

# COMMAND ----------

# Apply watermarks on event-time columns
stream_predict_watermarks = stream_predict.withWatermark("impressionTime", "2 hours")
stream_actual_watermark = stream_actual.withWatermark("actualTime", "3 hours")


some_model.join(stream_predict,
               expr(""" predictId = actualId AND
    actualTime >= impressionTime AND
    actualTime <= impressionTime + interval 1 hour
    """))