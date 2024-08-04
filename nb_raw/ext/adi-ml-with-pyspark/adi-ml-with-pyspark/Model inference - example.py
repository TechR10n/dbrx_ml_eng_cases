# Databricks notebook source
import mlflow
import mlflow.keras
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import col, struct

from tensorflow import keras
import tensorflow as tf

# COMMAND ----------

# MAGIC %md
# MAGIC  ### Move model from TesnorFlow to MLFlow registry

# COMMAND ----------

model_path =  "/home/jovyan/dist-tf-model/"

# COMMAND ----------

restored_keras_model = tf.keras.models.load_model(model_path)

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.keras.log_model(restored_keras_model, "models")

# COMMAND ----------

run_id ="425438f8a7b0471d9413684deeb63deb"
experiment_id = "0"

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions 
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .appName("Model inference") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define mlfloyw.pyfunc wrapper for the Model: 

# COMMAND ----------

# TIP: Create custom Python pyfunc model that transforms and predicts on inference data
# Allows the inference pipeline to be independent of the model framework used in training pipeline
class KerasCNNModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model_path):
    self.model_path = model_path

  def load_context(self, context):
    # Load the Keras-native representation of the MLflow
    # model
    print(self.model_path)
    self.model = mlflow.keras.load_model(
        model_uri=self.model_path)

  def predict(self, context, model_input):
    import tensorflow as tf
    import json

    class_def = {
      0: '212.teapot', 
      1: '234.tweezer', 
      2: '196.spaghetti', 
      3: '249.yo-yo', 
    }

    model_input['origin'] = model_input['origin'].str.replace("dbfs:","/dbfs")
    images = model_input['origin']

    rtn_df = model_input.iloc[:,0:1]
    rtn_df['prediction'] = None
    rtn_df['probabilities'] = None

    for index, row in model_input.iterrows():
      image = np.round(np.array(Image.open(row['origin']).resize((224,224)),dtype=np.float32))
      img = tf.reshape(image, shape=[-1, 224, 224, 3])
      class_probs = self.model.predict(img)
      classes = np.argmax(class_probs, axis=1)
      class_prob_dict = dict()
      for key, val in class_def.items():
        class_prob_dict[val] = np.round(np.float(class_probs[0][int(key)]), 3).tolist()
      rtn_df.loc[index,'prediction'] = classes[0]
      rtn_df.loc[index,'probabilities'] = json.dumps(class_prob_dict)

    return rtn_df[['prediction', 'probabilities']].values.tolist()


# COMMAND ----------

model_path = f"file:/home/jovyan/mlruns/{experiment_id}/{run_id}/artifacts/models"
wrappedModel = KerasCNNModelWrapper(model_path)
mlflow.pyfunc.log_model("pyfunc_model_v2", python_model=wrappedModel)
print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test the model with mlflow.pyfunc

# COMMAND ----------

# Test data. Using the same dataframe in this example
images_df = spark.read.parquet( "images_data/silver/augmented")
model_path = f"file:/home/jovyan/mlruns/{experiment_id}/{run_id}/artifacts/models"

# Always use the Production version of the model from the registry
mlflow_model_path = model_path

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, mlflow_model_path, result_type=ArrayType(StringType()))

# Predict on a Spark DataFrame.
scored_df = (images_df
             .withColumn('origin', col("content"))
             .withColumn('my_predictions', loaded_model(struct("origin")))
             .drop("origin"))


# COMMAND ----------

scored_df.show(5, truncate=False)

# COMMAND ----------

