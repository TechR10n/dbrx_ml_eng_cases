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
# MAGIC <h3>Use the Model Registry to manage the model lifecycle.</h3>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup
# MAGIC Before proceeding with this demonstration, you need to run the `Classroom-Setup` notebook to initialize the environment.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry Overview
# MAGIC
# MAGIC One of the primary challenges among data scientists and machine learning engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC
# MAGIC The [MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications
# MAGIC
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registering Models to Model Registry
# MAGIC
# MAGIC There are two ways to register models to Model Registry:
# MAGIC
# MAGIC 1. MLflow Run's page UI
# MAGIC 1. MLflow Model Registry API

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### MLflow Model Registry API
# MAGIC
# MAGIC Next, we'll provide the code to register a model and demonstrate how to request a stage transition.
# MAGIC
# MAGIC #### Registering a Model
# MAGIC
# MAGIC To register a model programmatically, you can use the following code block.
# MAGIC
# MAGIC Note that we're registering the same model to the same MLflow Model Registry Model here, so it'll create a new version of that model.

# COMMAND ----------

# Import libraries
import mlflow
from mlflow.tracking import MlflowClient

# Manually set parameter values
experiment_name = f"/Users/{DA.username}/databricks_automl/retail-prediction-experiment"
model_name = "retail-prediction-best-model"

# Instantiate client
client = MlflowClient()

# Find best run ID for experiment
experiment = client.get_experiment_by_name(experiment_name)
best_run_id = mlflow.search_runs(experiment.experiment_id).sort_values("metrics.val_r2_score", ascending=False).loc[0, "run_id"]

# Register model
model_uri = f"runs:/{best_run_id}/model"
model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Update Description
# MAGIC
# MAGIC We can now update the description with some useful information. This can be done both programatically and in the UI for the registered model.

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="This model predicts the number of units purchased by a customer."
)

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version has our best r^2 score yet."
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition to Staging
# MAGIC
# MAGIC Next, we can transition the model we just registered to "Staging".

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Staging",
    archive_existing_versions=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC And finally, head to the Model Registry to verify that your model is in the **Staging** stage.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>