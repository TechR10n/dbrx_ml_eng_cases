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
    .load("CO2_Emissions_Canada.csv")

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
data = hasher.transform(co2_data)
                       


# COMMAND ----------

data.select("hashed_features").show(5, truncate=False)

# COMMAND ----------

data.select("hashed_features").take(1)

# COMMAND ----------

data.select("hashed_features").show(5, truncate=False)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # time for selecting the most meaninful features:

# COMMAND ----------

from pyspark.ml.feature import UnivariateFeatureSelector

selector = UnivariateFeatureSelector(outputCol="selectedFeatures", featuresCol="hashed_features", labelCol="CO2")

selector.setFeatureType("continuous")
selector.setLabelType("continuous")

model = selector.fit(data)
data = model.transform(data)

# COMMAND ----------

# MAGIC %md
# MAGIC  ## Tryout LDA clustring algo

# COMMAND ----------

from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA


lda = LDA(k=2, seed=1, optimizer="em",featuresCol="selectedFeatures")
lda.setMaxIter(100)


lda.clear(lda.maxIter)
lda_model = lda.fit(data)
lda_model.setSeed(1)

# check if the model itself is distributed across Spark executres
lda_model.isDistributed()

# COMMAND ----------

lda_model.describeTopics().show()

# COMMAND ----------

lda_model.vocabSize()

# COMMAND ----------

lda_predictions = lda_model.transform(data)

# COMMAND ----------

lda_predictions.printSchema()

# COMMAND ----------

lda_predictions.select("topicDistribution").show(2,truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Tryout KMeans

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans



kmeans = KMeans(k=3)
kmeans.setSeed(10)
kmeans.setFeaturesCol("selectedFeatures")

kmeans_model = kmeans.fit(data)
kmeans_model.getDistanceMeasure()




# COMMAND ----------

kmeans_predictions = kmeans_model.transform(data)

# COMMAND ----------

kmeans_predictions.select("prediction").show(5, truncate=True)

# COMMAND ----------

kmeans_predictions.select("prediction").distinct().show()

# COMMAND ----------

summary = kmeans_model.summary

# COMMAND ----------

summary.cluster.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Tryout GaussianMixture

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture

gm = GaussianMixture(k=42, tol=0.01, seed=10, featuresCol="selectedFeatures", maxIter=100)
gm_model = gm.fit(data)

gm_predictions = gm_model.transform(data)

# COMMAND ----------

# MAGIC %md
# MAGIC Print the model params using `explainParams()` functionality:

# COMMAND ----------

import pprint
pp = pprint.PrettyPrinter(indent=4)
params = gm_model.explainParams()
pp.pprint(params)

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
# MAGIC ## Evaluating clustring models
# MAGIC
# MAGIC Notice we are not using this evaluator for LDA since it outputs topicDistribution and not one numeric prdiction.
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(featuresCol='selectedFeatures')
evaluator.setPredictionCol("prediction")

#evaluate with eucliden distance
print("kmeans: "+str(evaluator.evaluate(kmeans_predictions)))
print("GM: "+ str(evaluator.evaluate(gm_predictions)))

# COMMAND ----------

evaluator.isLargerBetter()

# COMMAND ----------

evaluator.setDistanceMeasure("cosine")
print("kmeans: "+str(evaluator.evaluate(kmeans_predictions)))
print("GM: "+ str(evaluator.evaluate(gm_predictions)))

# COMMAND ----------

evaluator.isLargerBetter()

# COMMAND ----------

evaluator.explainParams()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Since evaluator output for `isLargerBetter` was true, we can define that kmeans algorithm produced a better model than GM.

# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameters and Tuning experiments
# MAGIC

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit , ParamGridBuilder

grid = ParamGridBuilder().addGrid(kmeans.maxIter, [20,50,100]).build()

tvs = TrainValidationSplit(estimator=kmeans, estimatorParamMaps=grid, evaluator=evaluator,
                           collectSubModels=True, parallelism=1, seed=42)
tvs_model = tvs.fit(data)
tvs_model.getTrainRatio()



# COMMAND ----------

tvs_model.validationMetrics

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit , ParamGridBuilder

grid = ParamGridBuilder().addGrid(kmeans.maxIter, [20,50,100]) \
        .addGrid(kmeans.distanceMeasure, ['euclidean','cosine']).build()


# COMMAND ----------

tvs_model.validationMetrics

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit , ParamGridBuilder

grid = ParamGridBuilder().addGrid(kmeans.maxIter, [20,50,100]) \
        .addGrid(kmeans.distanceMeasure, ['euclidean','cosine']) \
        .addGrid(evaluator.distanceMeasure, ['euclidean','cosine']).build()


tvs = TrainValidationSplit(estimator=kmeans, estimatorParamMaps=grid, evaluator=evaluator,
                           collectSubModels=True, parallelism=1, seed=42, trainRatio=0.8)
tvs_model = tvs.fit(data)
tvs_model.validationMetrics



# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding evaluator to the grid params:

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit , ParamGridBuilder


grid = ParamGridBuilder().addGrid(kmeans.maxIter, [20,50,100]) \
        .addGrid(kmeans.distanceMeasure, ['euclidean','cosine']) \
        .addGrid(evaluator.distanceMeasure, ['euclidean','cosine'])\
        .baseOn({kmeans.featuresCol: 'selectedFeatures'}) \
        .build()

tvs = TrainValidationSplit(estimator=kmeans, estimatorParamMaps=grid, evaluator=evaluator,
                           collectSubModels=True, parallelism=1, seed=42, trainRatio=0.8)
tvs_model = tvs.fit(data)
tvs_model.validationMetrics





# COMMAND ----------

tvs_model.subModels

# COMMAND ----------

arr_models = tvs_model.subModels

# COMMAND ----------

# MAGIC %md
# MAGIC # Advanced Split
# MAGIC
# MAGIC the subModels are printed here as an example, do not use for real systems!!!

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, CrossValidatorModel


cv = CrossValidator(estimator=kmeans, estimatorParamMaps=grid, evaluator=evaluator,
                           collectSubModels=True,  parallelism=2, numFolds=3)

cv_model = cv.fit(data)
cv_model.subModels

# COMMAND ----------

len(cv_model.subModels)

# COMMAND ----------

len(cv_model.subModels[0])

# COMMAND ----------

cv_model.avgMetrics