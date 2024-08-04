# Databricks notebook source
# MAGIC %md
# MAGIC # Zoo Animal Clasification

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Use Machine Learning Methods to Correctly Classify Animals Based Upon Attributes.
# MAGIC Dataset by Kaggle. More information can be found [here](https://www.kaggle.com/uciml/zoo-animal-classification).

# COMMAND ----------

# MAGIC %md
# MAGIC # Provide custom schema for the data 

# COMMAND ----------


from pyspark.sql.types import StructField, StructType, StringType, DoubleType
# notice that although the most of the columns are of integet type, the custome schema will use integer type.
# This is because this is the statistic functionality expected numeric type. 

custom_schema = StructType([
    StructField("animal_name", StringType(), True),
    StructField("hair", DoubleType(), True),
    StructField("feathers", DoubleType(), True),
    StructField("eggs", DoubleType(), True),
    StructField("milk", DoubleType(), True),
    StructField("airborne", DoubleType(), True),
    StructField("aquatic", DoubleType(), True),
    StructField("predator", DoubleType(), True),
    StructField("toothed", DoubleType(), True),
    StructField("backbone", DoubleType(), True),
    StructField("breathes", DoubleType(), True),
    StructField("venomous", DoubleType(), True),
    StructField("fins", DoubleType(), True),
    StructField("legs", DoubleType(), True),
    StructField("tail", DoubleType(), True),
    StructField("domestic", DoubleType(), True),
    StructField("catsize", DoubleType(), True),
    StructField("class_type", StringType(), True)])

# COMMAND ----------

# load data
zoo_data_path = "dbfs:/FileStore/tmp/wkspc_admin/zoo.csv"
zoo_data = spark.read.format("csv")\
    .schema(custom_schema) \
    .option("header", True) \
    .load(zoo_data_path)

# COMMAND ----------

zoo_data.take(1)

# COMMAND ----------

zoo_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate statistics
# MAGIC for this, we will use the Summarizer functionality

# COMMAND ----------

# Statistic functionaly can only work on vector.
# Hence we will drop the columns of type string we dont need at the moment.

zoo_data_for_statistics = zoo_data.drop('animal_name','lass_type')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Turn the columns into a vector

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that for simplifying the example, we are going to examin the following columns:
# MAGIC
# MAGIC * feathers
# MAGIC * milk
# MAGIC * fins
# MAGIC * domestic 

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# use vector transformer as describe in the book under transofrmers in chapter 3
vecAssembler = VectorAssembler(outputCol="features")
# assemble only part of the columns for the example
vecAssembler.setInputCols(["feathers","milk","fins","domestic"])

vector_df = vecAssembler.transform(zoo_data_for_statistics)


# COMMAND ----------

vector_df.printSchema()

# COMMAND ----------

from pyspark.ml.stat import Summarizer
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

# create summarizer for multiple metrics "mean","variance","normL1","normL2","std" and "sum".
summarizer = Summarizer.metrics("mean","variance","normL1","normL2","std","sum","numNonZeros","max","min")


# compute statistics for multiple metrics with weight
statistics_df = vector_df.select(summarizer.summary(vector_df.features))

statistics_df.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC Notice that statistics dataframe has only one column named aggregate_metrics, where aggregate_metrics coluumns has more columns, where each one of them is a vector.

# COMMAND ----------

statistics_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC For enabling easier access to the data, we use explode functionality that flattens one hirarchy:

# COMMAND ----------

# compute statistics for single metric "std" without the rest
vector_df.select(Summarizer.std(vector_df.features)).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC From [wikipedia](https://en.wikipedia.org/wiki/Standard_deviation) std - Standard deviation is a measure of the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the values are spread out over a wider range. 
# MAGIC
# MAGIC Looking at the vector results, the distance from the among each individual feature is lower than 0.5
# MAGIC Our features: "feathers","milk","fins","domestic"
# MAGIC
# MAGIC The reson for it, mainly is, the data should be represented in boolean since each feature is a yes/no fearure.
# MAGIC Feathers =1 , means that this animal has feathers and so on.
# MAGIC
# MAGIC Now that we know this, let's take a look at count, which will tell us how many animals in the database has feathers, milk, fins or domestic.

# COMMAND ----------

# compute statistics for single metric "sum" without the rest
vector_df.select(Summarizer.sum(vector_df.features)).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC `sum` provides us with a more relatable information that we can use to understand the data. 

# COMMAND ----------

# compute statistics for single metric "variance" without the rest
vector_df.select(Summarizer.variance(vector_df.features)).show(truncate=False)

# COMMAND ----------

# compute statistics for single metric "count" without the rest
vector_df.select(Summarizer.count(vector_df.features)).show(truncate=False)

# COMMAND ----------

# compute statistics for single metric "numNonZeros" without the rest
vector_df.select(Summarizer.numNonZeros(vector_df.features)).show(truncate=False)

# COMMAND ----------

# compute statistics for single metric "max" without the rest
vector_df.select(Summarizer.max(vector_df.features)).show(truncate=False)

# COMMAND ----------

# compute statistics for single metric "normL1" without the rest
vector_df.select(Summarizer.normL1(vector_df.features)).show(truncate=False)

# COMMAND ----------

# compute statistics for single metric "normL2" without the rest
vector_df.select(Summarizer.normL2(vector_df.features)).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Testing features correlations
# MAGIC As part of understanding each featres statistics on its own, let's understand the correlation between the features. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notice
# MAGIC This functionality also requires a vector, we will use the one from the earlier computation - `vector_df`

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.stat import KolmogorovSmirnovTest

r1 = Correlation.corr(vector_df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0])+ "\n")

r2 = Correlation.corr(vector_df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC Breakdown of the correlation metrix is in the book, chapter 3 under statistics. 
# MAGIC

# COMMAND ----------

from pyspark.ml.stat import KolmogorovSmirnovTest

# COMMAND ----------

# MAGIC %md
# MAGIC ## ChiSquareTest
# MAGIC
# MAGIC Testing the p-value of the columns:

# COMMAND ----------

# MAGIC %md
# MAGIC This requeires vector as well Hence we will use the prcompute vector from before. 
# MAGIC
# MAGIC Notice that label in this case, has to be of type numberic.
# MAGIC To tranform the label into numberic, we will use the StringIndexer transofmer functionality

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="class_type", outputCol="label")
indexed_lable = indexer.fit(vector_df).transform(vector_df)


# COMMAND ----------

indexed_lable.printSchema()

# COMMAND ----------

indexed_lable.select("features").take(1)

# COMMAND ----------

indexed_lable

# COMMAND ----------


from pyspark.ml.stat import ChiSquareTest

chiSqResult = ChiSquareTest.test(indexed_lable, 'features', 'label')
chiSqResult.select("degreesOfFreedom").collect()[0]

# COMMAND ----------


chiSqResult = ChiSquareTest.test(indexed_lable, 'features', 'label', True)
row = chiSqResult.orderBy("featureIndex").collect()
row[0].statistic


# COMMAND ----------

row

# COMMAND ----------

# MAGIC %md
# MAGIC Reminder that for simplifying the example, we used the following columns:
# MAGIC * feathers
# MAGIC * milk
# MAGIC * fins
# MAGIC * domestic

# COMMAND ----------

