# Databricks notebook source
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

cols_only_continues_values = {'Fuel Consumption City (L/100 km)':0}
#                               "Fuel Consumption Hwy (L/100 km)",
#         "Fuel Consumption Comb (L/100 km)"}

# COMMAND ----------

co2_data = co2_data.fillna(0.0)

# COMMAND ----------

co2_data.printSchema()

# COMMAND ----------

co2_data.take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Prep the data for regression

# COMMAND ----------

# MAGIC %md
# MAGIC turn the feature columns into one indexed column:

# COMMAND ----------

from pyspark.ml.feature import FeatureHasher
from pyspark.sql.functions import col

cols = ["Make", "Model", "Vehicle Class","Cylinders","Transmission","Fuel Type",
        "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)",
        "Fuel Consumption Comb (L/100 km)","Fuel Consumption Comb (mpg)"]

cols_only_continues = ["Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)",
        "Fuel Consumption Comb (L/100 km)"]

hasher = FeatureHasher(outputCol="hashed_features", inputCols=cols_only_continues)
co2_data = hasher.transform(co2_data)
                       


# COMMAND ----------

co2_data.select("hashed_features").show(5, truncate=False)

# COMMAND ----------

co2_data.select("hashed_features").take(1)

# COMMAND ----------

co2_data.select("hashed_features").show(5, truncate=False)

# COMMAND ----------

co2_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # time for selecting the most meaninful features:

# COMMAND ----------

from pyspark.ml.feature import UnivariateFeatureSelector

selector = UnivariateFeatureSelector(outputCol="selectedFeatures", featuresCol="hashed_features", labelCol="CO2")

selector.setFeatureType("continuous")
selector.setLabelType("continuous")

model = selector.fit(co2_data)
output = model.transform(co2_data)

# COMMAND ----------

output.select("selectedFeatures").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Count the number of available classes as it is needed as part of the last layer for the multilayer perceptron classifier.
# MAGIC
# MAGIC Classifier trainer based on the Multilayer Perceptron. 
# MAGIC Each layer has sigmoid activation function, output layer has softmax. 
# MAGIC
# MAGIC Number of inputs has to be equal to the size of feature vectors. 
# MAGIC
# MAGIC Number of outputs has to be equal to the total number of labels.
# MAGIC
# MAGIC
# MAGIC Specify the layers for the neural network as follows: 
# MAGIC
# MAGIC input layer => size 50 (features), two intermediate layers (i.e. hidden layer) of size 20 and 8 and output => size 70 as the largest value of CO2 is 69 and lables array is searched by index (classes).      
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


mlp = MultilayerPerceptronClassifier(layers=[50,20, 8, 70], seed=123, featuresCol="selectedFeatures", labelCol="CO2")
mlp.setMaxIter(100)


model = mlp.fit(output)



# COMMAND ----------

