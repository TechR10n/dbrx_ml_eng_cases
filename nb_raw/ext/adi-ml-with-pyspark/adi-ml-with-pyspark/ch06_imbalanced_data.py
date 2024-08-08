# Databricks notebook source
from pyspark.sql import SparkSession 

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("Intro") \
    .getOrCreate()



# COMMAND ----------

df = spark.createDataFrame([
    (0, "Hi I think pyspark is cool ","happy"),
    (1, "All I want is a pyspark cluster","indifferent"),
    (2, "I finally understand how ML works","fulfill"),
    (3, "Yet another sentence about pyspark and ML","indifferent"),
    (4, "Why didn't I know about mllib before","sad"),
    (5, "Yes, I can","happy")
], ["id", "sentence", "sentiment"])

# COMMAND ----------

df

# COMMAND ----------

# MAGIC %md
# MAGIC # Start Featurization process

# COMMAND ----------

# MAGIC %md
# MAGIC # Leverage algo to target Imbalanced Data: 
# MAGIC Similar featurize process as before, only now we have one label for every data entry

# COMMAND ----------

# stop words:
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer

remover = StopWordsRemover()
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(df)

remover.setInputCol("words")
remover.setOutputCol("clean_words")

df = remover.transform(tokenized)

# COMMAND ----------

from pyspark.ml.feature import NGram

ngram = NGram(n=3, inputCol="clean_words", outputCol="ngrams")
test = ngram.transform(df)
test.show(5,truncate=False)

# COMMAND ----------

df = test

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline

hashtf = HashingTF(numFeatures=2**16, inputCol="ngrams", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=2) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "sentiment", outputCol = "label")

# example of how pipelines can help us orchestrate the featurize!
pipeline = Pipeline(stages=[hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(test)
train_df = pipelineFit.transform(test)
train_df.select("features","label").show(5,truncate=False)

# now our train data is ready to be used to produce classifier

# COMMAND ----------

# MAGIC %md
# MAGIC # Using GBTClassifier with specifiying strategy

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Train a RandomForestClassifier model with a dedicate feature subset strategy
rf = RandomForestClassifier(labelCol="label", featuresCol="features",featureSubsetStrategy="log2")
model = rf.fit(train_df)

# COMMAND ----------

# Make predictions.
predictions = model.transform(train_df)

# Select example rows to display.
predictions

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

predictions.select("rawPrediction","label","probability","prediction").show(5,truncate=True)

# COMMAND ----------

