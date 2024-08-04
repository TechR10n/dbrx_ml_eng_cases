# Databricks notebook source
# MAGIC %md
# MAGIC This notebook shows you how to create and query a table or DataFrame loaded from data stored in dataset.# Caltech 256 - Preprocess, processs and Feature Engineering
# MAGIC
# MAGIC 1. Init environment
# MAGIC 2. Load sample data
# MAGIC 3. Preprocess & Processes the data
# MAGIC 4. Feature engineeing
# MAGIC 7. Avoid small files when saving the data

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql.types import BinaryType,StringType
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Set the data location and type
# MAGIC
# MAGIC There are two ways to access Azure Blob storage: account keys and shared access signatures (SAS).
# MAGIC
# MAGIC To get started, we need to set the location and type of the file.

# COMMAND ----------

file_location = "256_sampledata/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Read the data
# MAGIC
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

! ls -l "256_sampledata"


# COMMAND ----------

# start Spark session:

spark = SparkSession \
    .builder \
    .appName("Marshalling Image data") \
    .config("spark.memory.offHeap.enabled",True) \
    .config("spark.memory.offHeap.size","30g")\
    .getOrCreate()

# COMMAND ----------

spark.sql("set spark.sql.files.ignoreCorruptFiles=true")

df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(file_location)



# COMMAND ----------

df.printSchema()

# COMMAND ----------

# Try image file type to learn about the schema:
# we are NOT using this DF.

image_df = spark.read.format("image") \
.option("pathGlobFilter", "*.jpg") \
.option("recursiveFileLookup", "true") \
.load(file_location)

# COMMAND ----------

image_df.printSchema()

# COMMAND ----------

image_df = None

# COMMAND ----------

df.show(5)

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## preprocess
# MAGIC  1. Extract labels
# MAGIC  2. Extract size
# MAGIC  3. transform labels to index

# COMMAND ----------

# MAGIC %md
# MAGIC #### Regex expression
# MAGIC Notice that every path file can be different, you will need to tweak the actual regex experssion to fit your file path. for that, take a look at an example of the file path and experiement with a [regex calculator](https://regexr.com/). 

# COMMAND ----------

df.select("path").show(5, truncate=False)

# COMMAND ----------


import io
import numpy as np
import pandas as pd
import uuid
from pyspark.sql.functions import col, pandas_udf, regexp_extract
from PIL import Image

def extract_label(path_col):
  """Extract label category number from file path using built-in sql function"""
  #([^/]+)
  return regexp_extract(path_col,"256_sampledata/([^/]+)",1)

def extract_size(content):
  """Extract images size from its raw content"""
  image = Image.open(io.BytesIO(content))
  return image.size

@pandas_udf("width: int, height: int")
def extract_size_udf(content_series):
  sizes = content_series.apply(extract_size)
  return pd.DataFrame(list(sizes))

# COMMAND ----------

images_w_label_size = df.select( 
    col("path"),
    extract_label(col("path")).alias("label"),
    extract_size_udf(col("content")).alias("size"),
    col("content"))

images_w_label_size.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC #Transform label to index

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1st way - the python way

# COMMAND ----------

labels = images_w_label_size.select(col("label")).distinct().collect()
label_to_idx = {label: index for index,(label,) in enumerate(sorted(labels))}
num_classes = len(label_to_idx)


@pandas_udf("long")
def get_label_idx(labels):
  return labels.map(lambda label: label_to_idx[label])

labels_idx = images_w_label_size.select( 
    col("label"),
    get_label_idx(col("label")).alias("label_index"),
    col("content"),
    col("path"),
    col("size"))

labels_idx.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2nd way - the mllib way

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed = indexer.fit(images_w_label_size).transform(images_w_label_size)

indexed.show(10)

# COMMAND ----------

indexed.select("label_index").distinct().collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3rd way - from the label itself

# COMMAND ----------

def extract_index_from_label(label):
  """Extract index  from label"""
  return regexp_extract(label,"^([^.]+)",1)

labels_idx = images_w_label_size.select( 
    col("label"),
    extract_index_from_label(col("label")).alias("label_index"),
    col("content"),
    col("path"),
    col("size"))

labels_idx.show(5,truncate=False)

# COMMAND ----------

images_w_label_size = None

# COMMAND ----------

df = indexed

# COMMAND ----------

labels_idx = None

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Feature Engineering
# MAGIC Extracting greyscale images.
# MAGIC Greyscale is used as an example of feature we might want to extract.

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### calculate average image size for each category
# MAGIC 1. flat the column into two columns
# MAGIC 2. calculate average size for category
# MAGIC 3. resize according to average.
# MAGIC
# MAGIC

# COMMAND ----------

# 1st step - flatten the struact 
flattened = df.withColumn('width', col('size')['width'])
flattened = flattened.withColumn('height', col('size')['height'])
flattened.select('width','height').show(3, truncate = False)

# COMMAND ----------

# 2 - calculate average size for category

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import Window

@pandas_udf("int")
def pandas_mean(size: pd.Series) -> (int):
    return size.sum()

flattened.select(pandas_mean(flattened['width'])).show()
flattened.groupby("label").agg(pandas_mean(flattened['width'])).show()
flattened.select(pandas_mean(flattened['width']).over(Window.partitionBy('label'))).show()


flattened.select(pandas_mean(flattened['height'])).show()
flattened.groupby("label").agg(pandas_mean(flattened['height'])).show()
flattened.select(pandas_mean(flattened['height']).over(Window.partitionBy('label'))).show()



# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract greyscale

# COMMAND ----------

# Sample python native function that can do additional processing - expects pandas df as input and returns pandas df as output.
def add_grayscale_img(input_df):
  # Set up return frame.  In this case I'll have a row per passed in row.  You could be aggregating down to a single image, slicing
  # out columns,or just about anything, here.  For this case, I am simply going to return the input_df with some extra columns.
  input_df['grayscale_image'] = input_df.content.apply(lambda image: get_image_bytes(Image.open(io.BytesIO(image)).convert('L'))) 
  input_df['grayscale_format'] = "png" # Since this is a pandas df, this will assigne png to all rows
  
  return input_df

def get_image_bytes(image):
  img_bytes = io.BytesIO()
  image.save(img_bytes,format="png")
  return img_bytes.getvalue()

# COMMAND ----------

# Setup the return schema. Add blank columns to match the schema expected after applying the transformation function. Makes the schema definition easy in the function invocation.
rtn_schema = (df.select('content','label','path')
                                 .withColumn('grayscale_image', lit(None).cast(BinaryType()))
                                 .withColumn('grayscale_format', lit(None).cast(StringType()))
               )
 

# COMMAND ----------

# Reduce df down to data used in the function, the groupBy, and the re-join key respectively.  This could include other features as used by your pandas function
limited_df = df.select('label','content','path')

# COMMAND ----------

# Returns spark dataframe with transformations applied in parallel for each 'group'
augmented_df = limited_df.groupBy('label').applyInPandas(add_grayscale_img, schema=rtn_schema.schema)

# COMMAND ----------

# re-join to the full dataset using leftouter in case the image transform needed to skip some rows
output_df = df.join(augmented_df.select('path','grayscale_image'),['path'],"leftouter")      

# COMMAND ----------

# MAGIC %md
# MAGIC # Test on small data

# COMMAND ----------

pd_df = limited_df.limit(5).toPandas()
print(pd_df.columns)

# COMMAND ----------

limited_df = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make sure function works correctly

# COMMAND ----------


# Some testing code
test_df = pd_df.copy()
add_grayscale_img(test_df)
print(test_df['grayscale_image'])


# COMMAND ----------


from PIL import ImageFilter
# Sample python native function that can do additional processing - expects pandas df as input and returns pandas df as output.
def add_laplas(input_df):
  # Set up return frame.  In this case I'll have a row per passed in row.  You could be aggregating down to a single image, slicing
  # out columns,or just about anything, here.  For this case, I am simply going to return the input_df with some extra columns.
  input_df['edges_image'] = input_df.grayscale_image.apply(lambda image: get_image_bytes(Image.open(io.BytesIO(image)).filter(ImageFilter.FIND_EDGES)
)) 
  return input_df



# COMMAND ----------

# Some testing code
add_laplas(test_df)
print(test_df['edges_image'])

# COMMAND ----------

print(test_df['path'][4])

# COMMAND ----------

test_df

# COMMAND ----------

print(test_df.columns)

# COMMAND ----------

# display one image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

color_image = mpimg.imread(io.BytesIO(test_df.loc[1,'content']), format='jpg')
image = mpimg.imread(io.BytesIO(test_df.loc[1,'grayscale_image']), format='png')
edges_image = mpimg.imread(io.BytesIO(test_df.loc[1,'edges_image']), format='png')
print('color dimensions = {}'.format(color_image.shape))
print('grayscale dimensions = {}'.format(image.shape))

row_count = test_df.count()[0]
plt.figure(figsize=(8,20))
for label_index,row in test_df.iterrows():
  (_,content,_,grayscale,_,_) = row
  color_image = mpimg.imread(io.BytesIO(content), format='jpg')
  image = mpimg.imread(io.BytesIO(grayscale), format='png')

  plt.subplot(row_count,2,label_index*2+1)
  plt.imshow(color_image)
  plt.subplot(row_count,2,label_index*2+2)
  plt.imshow(image,cmap='gray')


# COMMAND ----------

#laplas kernel convolution
plt.figure(figsize=(8,20))
for label_index,row in test_df.iterrows():
  (_,content,_,grayscale,_,edges_image) = row
  edges_image = image = mpimg.imread(io.BytesIO(edges_image), format='png')
    
  plt.subplot(row_count,1,label_index*1+1)
  plt.imshow(edges_image,cmap='gray')

# COMMAND ----------

# MAGIC %md
# MAGIC # Full Dataset

# COMMAND ----------

output_df.show(2, truncate=True)

# COMMAND ----------

output_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5: scale the image
# MAGIC
# MAGIC From the size column, we notice that caltech_256 image size highly varay. To proced with the process, we need to scale the images to have a unannimous size. For tha we will use Spark UDFs with PIL.
# MAGIC
# MAGIC This is a must do part of normalizing and preprocessing image data.

# COMMAND ----------

from pyspark.sql.types import BinaryType, IntegerType
from pyspark.sql.functions import udf

img_size = 224

def scale_image(image_bytes):
    try:
      image = Image.open(io.BytesIO(image_bytes)).resize([img_size, img_size])
      return image.tobytes()
    except:
        return None

# COMMAND ----------

array = output_df.select("content").take(1)

# COMMAND ----------

from pyspark.sql.functions import udf
scale_image_udf = udf(scale_image, BinaryType())

# COMMAND ----------

#image_df = output_df.select("label_index", scale_image_udf("content").alias("content"))
image_df = output_df.select("label_index", scale_image_udf(col("content")).alias("image"))

# COMMAND ----------

image_df.printSchema()

# COMMAND ----------

image_df = image_df.select("label_index","image",col("image").alias("content"))
image_df.printSchema()

# COMMAND ----------

image_df =image_df.drop("image")
image_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4: Save and Avoid small files problem
# MAGIC Save the image data into a file format where you can query and process at scale
# MAGIC
# MAGIC Saving the dataset with the greyscale.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Repartition and save to **parquet**

# COMMAND ----------

# incase you are running on a distributed environment, with a large dataset, it's a good idea to partition t

# save the data:

save_path_augmented = "images_data/silver/augmented"
# Images data is already compressed so we turn off parquet compression
compression = spark.conf.get("spark.sql.parquet.compression.codec")
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")



# COMMAND ----------

output_df.write.mode("overwrite").parquet(save_path_augmented)


# COMMAND ----------

save_path_filtered = "images_data/silver/filtered"
# parquet.block.size is for Petastorm, later
image_df.repartition(2).write.mode("overwrite").option("parquet.block.size", 1024 * 1024).parquet(save_path_filtered)

# COMMAND ----------

spark.conf.set("spark.sql.parquet.compression.codec", compression)

# COMMAND ----------

