# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting Started with Databricks ML
# MAGIC <h2>Module 4: Managing the model life cycle with Databricks ML</h2>
# MAGIC <h3>Deploying a model for batch inference.</h3>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup
# MAGIC Before proceeding with this demonstration, you need to run the `Classroom-Setup` notebook to initialize the environment.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Loading Necessary Components
# MAGIC
# MAGIC In this demonstration, we'll be deploying a model for batch inference.
# MAGIC
# MAGIC Before we begin, we need to load the **model** and the **feature table**.
# MAGIC
# MAGIC ### Load Model
# MAGIC
# MAGIC We can load the model directly from the Model Registry.
# MAGIC
# MAGIC Note that when we load the model as a Spark UDF, it allows us to easily broadcast the model to the different nodes in our cluster to scale out our inference on our distributed feature table.

# COMMAND ----------

import mlflow

model = mlflow.pyfunc.spark_udf(spark, model_uri="models:/retail-prediction-best-model/production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Feature Table
# MAGIC
# MAGIC Next, we need to load our feature table. Remember that we created this feature table way back in our first demonstration.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
features = fs.read_table(f"{DA.schema_name}.customer_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performing Batch Inference
# MAGIC
# MAGIC In order to perform batch inference, 

# COMMAND ----------

predictions = features.withColumn('predictions', model(*features.columns))
display(predictions.select("units_purchased", "predictions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Lake
# MAGIC
# MAGIC Finally, we can write our features to Delta Lake.

# COMMAND ----------

predictions.write.mode("append").saveAsTable(f"{DA.schema_name}.predictions")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>