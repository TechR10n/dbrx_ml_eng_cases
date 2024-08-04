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
# MAGIC <h3>Automatic Model Retraining.</h3>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup
# MAGIC Before proceeding with this demonstration, you need to run the `Classroom-Setup` notebook to initialize the environment.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Scheduling a machine learning training workflow
# MAGIC
# MAGIC In this notebook, we'll create a workflow to retrain our model. Then, we'll set up this notebook to run weekly using a Databricks Job to ensure our model is always up-to-date.
# MAGIC
# MAGIC ### Load Features
# MAGIC
# MAGIC First, we'll load in our feature table.
# MAGIC
# MAGIC Note that in this demonstration, we are using the same records &mdash; but in real-world scenario, we'd likely have updated records appended to this table each time the model is trained.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

feature_table = f"{DA.schema_name}.customer_features"
fs = FeatureStoreClient()
features = fs.read_table(feature_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML Process
# MAGIC
# MAGIC Next, we'll use the AutoML API to kick off an AutoML regression experiment. This is similar to what we did with the AutoML UI, but we can use the API to automate this process.

# COMMAND ----------

import databricks.automl
model = databricks.automl.regress(
    features, 
    target_col="units_purchased",
    primary_metric="r2",
    timeout_minutes=5,
    
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Best Model
# MAGIC
# MAGIC Once the AutoML experiment is done, we can identify the best model from the experiment and register that model to the Model Registry.

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id
model_name = "gdml-unit-prediction"
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Request Transition to Staging
# MAGIC
# MAGIC Once the model is registered, we request that it be transitioned to the **Staging** stage for testing.

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll use MLflow Client's `transition_model_version_stage` method to transition our model from `Staging` to `Production`.

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built programmatically using AutoML"
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Scheduling the Training Workflow
# MAGIC
# MAGIC Now that we've created our training workflow, we're going to schedule this notebook to run as a Job in a Workflow. To do this, navigate to Workflows in the menu on the left.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>