# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 5: Bot-Or-Not feature Extraction
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession 
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.fpm import FPGrowth, FPGrowthModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Tokenizer

# COMMAND ----------

bot_csv_path = "dbfs:/FileStore/tmp/wkspc_admin/bots_data.csv"
not_bot_csv_path = "dbfs:/FileStore/tmp/wkspc_admin/nonbots_data.csv"

df = spark.read.format("csv")\
    .option("header", "true") \
    .option("inferschema", "true") \
    .load(bot_csv_path)
df.display()

# COMMAND ----------

df = df.select("description")
df.display()

# COMMAND ----------

import pyspark.sql.functions as fn
df = df.withColumn(fn.col("description")).fn.encode(fn.col("description"), 'utf-8')
df.display()

# COMMAND ----------

df_not_bot = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferschema", "true") \
    .load(not_bot_csv_path) \
    .select("description")
df_not_bot.display()

# COMMAND ----------

df_train = spark.read.parquet("classified_train_data")

# COMMAND ----------

df_train = df_train.fillna({'bot':0})

# COMMAND ----------

data = spark.read.parquet('train_data_only_description')
data = data.fillna({'label':0})

# COMMAND ----------

tokenizer = Tokenizer(inputCol="description", outputCol="words")
wordsData = tokenizer.transform(data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Feature Extraction
# MAGIC ## TF-IDF feature extraction
# MAGIC Leveraging text technique to try and extract meaningful features 

# COMMAND ----------

wordsData.select('label').distinct().show()

# COMMAND ----------

# first TF using HashingTF. alternatively, we can use CountVectorizer to get term frequency vectors

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

hashingTF = HashingTF(inputCol="words", outputCol="frequencyFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

featurizedData.select("label", "frequencyFeatures").show(5)

# COMMAND ----------

# second IDF model

idf = IDF(inputCol="frequencyFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("label", "features").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## N-Gram feature extraction

# COMMAND ----------

from pyspark.ml.feature import NGram

ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

ngramDataFrame = ngram.transform(wordsData)
ngramDataFrame.select("ngrams").show(truncate=False)

# COMMAND ----------

ngram.explainParams()

# COMMAND ----------

