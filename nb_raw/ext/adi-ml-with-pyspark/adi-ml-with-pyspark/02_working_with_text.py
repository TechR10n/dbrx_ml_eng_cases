# Databricks notebook source
# MAGIC %md
# MAGIC ## Working with Text Data

# COMMAND ----------

# MAGIC %md
# MAGIC Define a documentDB

# COMMAND ----------

from pyspark.ml.feature import Word2Vec

# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)

result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

sentenceDataFrame = spark.createDataFrame([
    (0, "Hi|I|heard|about|Spark"),
    (1, "I     wish Java      could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["id", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
# alternatively, pattern="\\w+", gaps(False)

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("sentence", "words")\
    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("sentence", "words") \
    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

sentenceData = spark.createDataFrame([
    (0, ["I", "saw", "the", "red", "balloon"]),
    (1, ["Mary", "had", "a", "little", "lamb"])
], ["id", "raw"])

remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
remover.transform(sentenceData).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Code Example -> Tokanizer -> N Gram

# COMMAND ----------

from pyspark.ml.feature import NGram
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark "),
    (1, "I wish, wish Java, Java could"),
    (2, "Logistic regression, regression models")
], ["id", "sentence"])


tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
countTokens = udf(lambda words: len(words), IntegerType())
wordDataFrame = tokenizer.transform(sentenceDataFrame)

ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.select("ngrams").show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import Binarizer

continuousDataFrame = spark.createDataFrame([
    (0, 5.1),
    (1, 5.8),
    (2, 0.2)
], ["id", "feature"])

binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

binarizedDataFrame = binarizer.transform(continuousDataFrame)

print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
binarizedDataFrame.show()

# COMMAND ----------

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (Vectors.dense([2.0, 1.0]),),
    (Vectors.dense([0.0, 0.0]),)
], ["features"])

polyExpansion = PolynomialExpansion(degree=5, inputCol="features", outputCol="polyFeatures")
polyDF = polyExpansion.transform(df)

polyDF.show(truncate=False)


# COMMAND ----------

from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
    (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")

dctDf = dct.transform(df)

dctDf.select("featuresDCT").show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.linalg import Vectors

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -8.0,200]),),
    (1, Vectors.dense([2.0, 1.0, -4.0,2]),),
    (2, Vectors.dense([4.0, 10.0, 8.0,0]),)
], ["id", "features"])

scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(dataFrame)

# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(dataFrame)

scaledData.select("features", "scaledFeatures").show()

# COMMAND ----------

from pyspark.ml.feature import Bucketizer

splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]

data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
dataFrame = spark.createDataFrame(data, ["features"])

bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")

# Transform original data into its bucket index.
bucketedData = bucketizer.transform(dataFrame)

print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits())-1))
bucketedData.show()

# COMMAND ----------

from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors

# Create some vector data; also works for sparse vectors
data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
df = spark.createDataFrame(data, ["vector"])
transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),
                                 inputCol="vector", outputCol="transformedVector")
# Batch transform the vectors to create new column:
transformer.transform(df).show()

# COMMAND ----------

from pyspark.ml.feature import Imputer

df = spark.createDataFrame([
    (1.0, float("nan")),
    (2.0, float("nan")),
    (float("nan"), 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
], ["a", "b"])

imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
model = imputer.fit(df)

model.transform(df).show()

# COMMAND ----------

sentence_data_frame = spark.createDataFrame([
    (0, "Hi I think pyspark is cool ","happy"),
    (1, "All I want is a pyspark cluster","indifferent"),
    (2, "I finally understand how ML works","Fulfilled"),
    (3, "Yet another sentence about pyspark and ML","indifferent"),
    (4, "Why didn't I know about mllib before","sad"),
    (5, "Yes, I can","happy")
], ["id", "sentence", "sentiment"])



# COMMAND ----------

sentence_data_frame

# COMMAND ----------

from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

tokenized = tokenizer.transform(sentence_data_frame)
tokenized.show(truncate=False)


# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")
meaningful_data_frame = remover.transform(tokenized)

meaningful_data_frame.select("words","meaningful_words").show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="sentiment", outputCol="categoryIndex")
indexed = indexer.fit(meaningful_data_frame).transform(meaningful_data_frame)
indexed.show()

# COMMAND ----------

indexed

# COMMAND ----------

sentiment_data_frame = spark.createDataFrame([
    (0, 0.01,0.43,0.3,0.5),
    (1, 0.097,0.21,0.2,0.9),
    (2, 0.4,0.329,0.97,0.4),
    (3, 0.7,0.4,0.3,0.87),
    (4, 0.34,0.4,0.3,0.78),
    (5, 0.1,0.3,0.31,0.29)
], ["sentence_id", "happy", "indifferent","Fulfilled","sad"])

sentiment_data_frame.show()

# COMMAND ----------

casted_data_frame = sentiment_data_frame.selectExpr("cast(happy as double)")
casted_data_frame.show()

# COMMAND ----------

sentiment_data_frame.printSchema()

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.show()

# COMMAND ----------

