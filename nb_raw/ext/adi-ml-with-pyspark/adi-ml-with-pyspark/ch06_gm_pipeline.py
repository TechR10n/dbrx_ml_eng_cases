# Databricks notebook source
# MAGIC %md
# MAGIC # Clustering CO2 dataset with GaussianMixture
# MAGIC Use Machine Learning Methods to cluster cars and their CO2 emission. 
# MAGIC Dataset by Kaggle. More information can be found [here](https://www.kaggle.com/debajyotipodder/).

# COMMAND ----------

from pyspark.sql import SparkSession 

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("Intro") \
    .getOrCreate()



# COMMAND ----------

from pyspark.sql.types import StructField, StructType, StringType, DoubleType

custom_schema = StructType([
    StructField("Make", StringType(), True),
    StructField("Model", StringType(), True),
    StructField("Vehicle Class", StringType(), True),
    StructField("Cylinders", DoubleType(), True),
    StructField("Transmission", StringType(), True),
    StructField("Fuel Type", StringType(), True),
    StructField("Fuel Consumption City (L/100 km)", DoubleType(), True),
    StructField("Fuel Consumption Hwy (L/100 km)", DoubleType(), True),
    StructField("Fuel Consumption Comb (L/100 km)", DoubleType(), True),
    StructField("Fuel Consumption Comb (mpg)", DoubleType(), True),
    StructField("CO2", DoubleType(), True)])


# COMMAND ----------

# load data

co2_data = spark.read.format("csv")\
    .schema(custom_schema) \
    .option("header", True) \
    .load("../datasets/CO2_Emissions_Canada.csv")

# COMMAND ----------

co2_data.take(2)

# COMMAND ----------

co2_data = co2_data.fillna(0.0)

# COMMAND ----------

co2_data.printSchema()

# COMMAND ----------

co2_data.take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Build Hasher

# COMMAND ----------

# MAGIC %md
# MAGIC turn the feature columns into one indexed column:

# COMMAND ----------

from pyspark.ml.feature import FeatureHasher
from pyspark.sql.functions import col


cols_only_continues = ["Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)",
        "Fuel Consumption Comb (L/100 km)"]

hasher = FeatureHasher(outputCol="hashed_features", inputCols=cols_only_continues)
                   

# COMMAND ----------

# MAGIC %md
# MAGIC # Build Selector

# COMMAND ----------

from pyspark.ml.feature import UnivariateFeatureSelector

selector = UnivariateFeatureSelector(outputCol="selectedFeatures", featuresCol="hashed_features", labelCol="CO2")

selector.setFeatureType("continuous")
selector.setLabelType("continuous")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create GaussianMixture

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture

gm = GaussianMixture(k=42, tol=0.01, seed=10, featuresCol="selectedFeatures", maxIter=100)


# COMMAND ----------

# MAGIC %md
# MAGIC # Constructing - The Pipeline API

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[hasher,selector, gm])
# Fit the pipeline to training data.
pipeline_model = pipeline.fit(co2_data)

# COMMAND ----------

transformed_by_pipeline = pipeline_model.transform(co2_data)

# COMMAND ----------

transformed_by_pipeline.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Persisting the pipeline to disk

# COMMAND ----------

path_model_with_pip = "/tmp/pip_model"
pipeline_model.write().overwrite().save(path_model_with_pip)


# COMMAND ----------

# MAGIC %md
# MAGIC # Using our model in Stream processing:

# COMMAND ----------

# assume we have data ingested in stream into our system:
data_in_stream = spark \
    .readStream \
    .schema(custom_schema) \
    .format("csv")\
    .option("header", True) \
    .load("StreamData/")

# COMMAND ----------

from pyspark.ml import PipelineModel

pipeline_from_disk = PipelineModel.load(path_model_with_pip)

# COMMAND ----------

from pyspark.sql.functions import when, col, sum

transformed_output = pipeline_from_disk.transform(data_in_stream)\
  .agg((sum(when(col('prediction') == 1, 1))))

# COMMAND ----------

transformed_output

# COMMAND ----------

query = transformed_output.writeStream.outputMode('complete').queryName("spark_streaming_ml").format('memory').start()


# COMMAND ----------

query.explain()

# COMMAND ----------

query.awaitTermination(20)

# COMMAND ----------

from pyspark import sql

output = spark.sql("select * from spark_streaming_ml")

# COMMAND ----------

output.show()

# COMMAND ----------

