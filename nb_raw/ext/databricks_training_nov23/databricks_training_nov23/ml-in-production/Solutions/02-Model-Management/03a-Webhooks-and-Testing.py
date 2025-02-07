# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-8a133a94-0841-46ab-ad14-c85a27948a3c
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC # MLflow Webhooks & Testing
# MAGIC
# MAGIC Webhooks trigger the execution of code (oftentimes tests) upon some event. This lesson explores how to employ webhooks to trigger automated tests against models in the model registry. 
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Explore the role of webhooks in ML pipelines
# MAGIC  - Create a job to test models in the model registry
# MAGIC  - Automate that job using MLflow webhooks
# MAGIC  - Create a HTTP webhook to send notifications to Slack

# COMMAND ----------

# DBTITLE 0,--i18n-5b114129-189d-4e53-a066-d3ccfbf8416c
# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **12.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 0,--i18n-1889b8f8-0c82-4057-892b-fe894ba35f18
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Automated Testing
# MAGIC
# MAGIC The backbone of the continuous integration, continuous deployment (CI/CD) process is the automated building, testing, and deployment of code. A **webhook or trigger** causes the execution of code based upon some event.  This is commonly when new code is pushed to a code repository.  In the case of machine learning jobs, this could be the arrival of a new model in the model registry.
# MAGIC
# MAGIC The two types of <a href="https://docs.databricks.com/applications/mlflow/model-registry-webhooks.html" target="_blank">**MLflow Model Registry Webhooks**</a>:
# MAGIC  - Webhooks with Job triggers: Trigger a job in a Databricks workspace
# MAGIC  - Webhooks with HTTP endpoints: Send triggers to any HTTP endpoint
# MAGIC  
# MAGIC This lesson uses:
# MAGIC 1. a **Job webhook** to trigger the execution of a Databricks job 
# MAGIC 2. a **HTTP webhook** to send notifications to Slack 
# MAGIC
# MAGIC Upon the arrival of a new model version with a given name in the model registry, the function of the Databricks job is to:<br><br>
# MAGIC - Import the new model version
# MAGIC - Test the schema of its inputs and outputs
# MAGIC - Pass example code through the model
# MAGIC
# MAGIC This covers many of the desired tests for ML models.  However, throughput testing could also be performed using this paradigm. Also, the model could also be promoted to the production stage in an automated fashion.

# COMMAND ----------

pip install databricks-registry-webhooks

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# DBTITLE 0,--i18n-35f9b2a0-9429-4922-b14d-cccceb711222
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create a Model and Job
# MAGIC
# MAGIC The following steps will create a Databricks job using another notebook in this directory: **`03b-Webhooks-Job-Demo`**

# COMMAND ----------

# DBTITLE 0,--i18n-bae3f327-645e-48fb-b920-a87d2a5cda23
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Create a user access token
# MAGIC
# MAGIC Create a user access token using the following steps:<br><br>
# MAGIC
# MAGIC 1. Click the Settings icon
# MAGIC 1. Click User Settings
# MAGIC 1. Go to the Access Tokens tab
# MAGIC 1. Click the Generate New Token button
# MAGIC 1. Optionally enter a description (comment) and expiration period
# MAGIC 1. Click the Generate button
# MAGIC 1. Copy the generated token **and paste it in the following cell**
# MAGIC
# MAGIC **Note:**
# MAGIC * Ensure that you are an admin on this workspace and that you're not using Community Edition (which has jobs disabled). 
# MAGIC * If you are not an admin, ask the instructor to share their token with you. 
# MAGIC * Alternatively, you can set **`token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()`**. However, this is not a best practice. We recommend you create your personal access token using the steps above and save it in your [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html). 
# MAGIC
# MAGIC
# MAGIC You can find details <a href="https://docs.databricks.com/dev-tools/api/latest/authentication.html" target="_blank">about access tokens here</a>

# COMMAND ----------

# ANSWER
# Programmatically get the token
import mlflow

token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()

# COMMAND ----------

import mlflow

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

instance = mlflow.utils.databricks_utils.get_webapp_url()

# COMMAND ----------

# DBTITLE 0,--i18n-086b4385-9eae-492e-8ccd-52d68a97ad86
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Train and Register a Model
# MAGIC
# MAGIC Build and log your model.

# COMMAND ----------

from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="Webhook RF Experiment") as run:
    # Data prep
    df = pd.read_parquet(f"{DA.paths.datasets_path}/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
    signature = infer_signature(X_train, pd.DataFrame(y_train))
    example = X_train.head(3)

    # Train and log model
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    mlflow.sklearn.log_model(rf, "random-forest-model", signature=signature, input_example=example)
    mse = mean_squared_error(y_test, rf.predict(X_test))
    mlflow.log_metric("mse", mse)
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

# COMMAND ----------

# DBTITLE 0,--i18n-8f56343e-2a5f-4515-be64-047b07dcf877
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Register the model

# COMMAND ----------

suffix = DA.unique_name("-")
name = f"webhook-demo_{suffix}"
model_uri = f"runs:/{run_id}/random-forest-model"

model_details = mlflow.register_model(model_uri=model_uri, name=name)

# COMMAND ----------

# DBTITLE 0,--i18n-02c615b7-dbf6-4e4a-8706-6c31cac2be68
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Creating the Job
# MAGIC
# MAGIC The following steps will create a Databricks job using another notebook in this directory: **`03b-Webhooks-Job-Demo`**

# COMMAND ----------

# DBTITLE 0,--i18n-b22313af-97a9-43d8-aaf6-57755b3d45da
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Create a job that executes the notebook **`03b-Webhooks-Job-Demo`** in the same folder as this notebook.<br><br>
# MAGIC
# MAGIC - Hover over the sidebar in the Databricks UI on the left.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/workflows-sidebar-job-v2.png" alt="step12" width="150"/>
# MAGIC <br></br>
# MAGIC
# MAGIC - Click on Create Job
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/create-job-button-v2.png" alt="step12" width="750"/>
# MAGIC
# MAGIC <br></br>
# MAGIC - Name your Job
# MAGIC - Select the notebook **`03b-Webhooks-Job-Demo`** 
# MAGIC - Select the current cluster
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/create-job-steps-v2.png" alt="step12" width="1000"/>
# MAGIC
# MAGIC <br></br>
# MAGIC - Copy the Job ID
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/job-details-id-v2.png" alt="step12" width="1000"/>

# COMMAND ----------

# DBTITLE 0,--i18n-66dd2af1-92c3-406c-8fea-4d755700cd73
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Alternatively, the code below will programmatically create the job.

# COMMAND ----------

import requests

def find_job_id(instance, headers, job_name, offset_limit=1000):
    params = {"offset": 0}
    uri = f"{instance}/api/2.1/jobs/list"
    res = requests.get(uri, params=params, headers=headers)
    
    if res.status_code == 200:
        job_list = res.json().get("jobs", [])
        if len(job_list) > 0:
            return job_list[0]["job_id"] ## return the first matching job
    else:
        None

def get_job_parameters(job_name, cluster_id, notebook_path):
    params = {
            "name": job_name,
            "tasks": [{"task_key": "webhook_task", 
                       "existing_cluster_id": cluster_id,
                       "notebook_task": {
                           "notebook_path": notebook_path
                       }
                      }]
        }
    return params

def get_create_parameters(job_name, cluster_id, notebook_path):
    api = "api/2.1/jobs/create"
    return api, get_job_parameters(job_name, cluster_id, notebook_path)

def get_reset_parameters(job_name, cluster_id, notebook_path, job_id):
    api = "api/2.1/jobs/reset"
    params = {"job_id": job_id, "new_settings": get_job_parameters(job_name, cluster_id, notebook_path)}
    return api, params

def get_webhook_job(instance, headers, job_name, cluster_id, notebook_path):
    job_id = find_job_id(instance, headers, job_name)
    if job_id is None:
        api, params = get_create_parameters(job_name, cluster_id, notebook_path)
    else:
        api, params = get_reset_parameters(job_name, cluster_id, notebook_path, job_id)
    
    uri = f"{instance}/{api}"
    res = requests.post(uri, headers=headers, json=params)
    assert res.status_code == 200, f"Expected an HTTP 200 response, received {res.status_code}; {res.content}"
    job_id = res.json().get("job_id", job_id)
    return job_id

notebook_path = mlflow.utils.databricks_utils.get_notebook_path().replace("03a-Webhooks-and-Testing", "03b-Webhooks-Job-Demo")

# We can use our utility method for creating a unique 
# database name to help us construct a unique job name.
prefix = DA.unique_name("-")
job_name = f"{prefix}_webhook-job"

# if the Job was created via UI, set it here.
job_id = get_webhook_job(instance, 
                         headers, 
                         job_name,
                         spark.conf.get("spark.databricks.clusterUsageTags.clusterId"),
                         notebook_path)

print(f"Job ID:   {job_id}")
print(f"Job name: {job_name}")

# COMMAND ----------

# DBTITLE 0,--i18n-15eeee0f-9472-4b7b-94b9-e728a7c9c38d
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Examine the Job
# MAGIC
# MAGIC Take a look at [the notebook you just scheduled]($./03b-Webhooks-Job-Demo) to see what it accomplishes.

# COMMAND ----------

# DBTITLE 0,--i18n-8c0aa70d-ab84-4e31-9ee4-2ab5d9fa6beb
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create a Job Webhook
# MAGIC
# MAGIC There are a few different events that can trigger a Webhook. In this notebook, we will be experimenting with triggering a job when our model transitions between stages.

# COMMAND ----------

from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec 

job_spec = JobSpec(
    job_id=job_id, 
    workspace_url=instance,
    access_token=token 
)
job_webhook = RegistryWebhooksClient().create_webhook(
    model_name=name, 
    events=["MODEL_VERSION_TRANSITIONED_STAGE"], 
    job_spec=job_spec,
    description="Job webhook trigger",
    status="ACTIVE" 
)

# COMMAND ----------

# DBTITLE 0,--i18n-965cfc78-c346-40d2-a328-d3d769a8c3e2
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now that we have registered the webhook, we can **test it by transitioning our model from stage `None` to `Staging` in the Experiment UI.** We should see in the Jobs tab that our Job has run.

# COMMAND ----------

# DBTITLE 0,--i18n-dc8d88f1-f954-4cbf-86cd-4e4c13c198db
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC To get the active Webhook related to our model, use `list_webhooks` and specify the `model_name`.

# COMMAND ----------

webhooks_list = RegistryWebhooksClient().list_webhooks(model_name=name)
current_job_webhook_id = job_webhook.id
print(webhooks_list)

# COMMAND ----------

# DBTITLE 0,--i18n-70e6903f-0d4e-423d-b4f8-aea6efd28ba5
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC Finally, delete the webhook by copying the webhook ID to the curl or python request. You can confirm that the Webhook was deleted by using the list request.

# COMMAND ----------

# ANSWER
http_webhook = RegistryWebhooksClient().delete_webhook(  
    id=current_job_webhook_id
)

# COMMAND ----------

webhooks_list = RegistryWebhooksClient().list_webhooks(model_name=name)
print(webhooks_list)

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
# MAGIC %md
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# DBTITLE 0,--i18n-d9dbc19c-7c1d-4cae-a198-5c6266acf825
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC - See this <a href="https://www.databricks.com/blog/2022/02/01/streamline-mlops-with-mlflow-model-registry-webhooks.html" target="_blank">blog</a> for more details streamlining MLOps with MLflow Model Registry Webhooks
# MAGIC - Set a Slack incoming webhook following <a href="https://api.slack.com/messaging/webhooks" target="_blank">this page</a>.
# MAGIC - Search the Databricks documentation for more details on <a href="https://docs.databricks.com/mlflow/model-registry-webhooks.html#examples" target="_blank"> MLflow webhooks</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>