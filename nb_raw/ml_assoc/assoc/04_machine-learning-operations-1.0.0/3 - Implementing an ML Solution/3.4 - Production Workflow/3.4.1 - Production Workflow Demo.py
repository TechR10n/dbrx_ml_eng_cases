# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Production Workflow Demo
# MAGIC
# MAGIC In this demo, we will provide a walkthrough of the production workflow for machine learning projects according to MLOps best practices using MLOps Stacks. We'll also address additional topics related to running solutions in production: like monitoring, model retraining, productionization, and compute resource configuration.
# MAGIC
# MAGIC **Note**: This demonstration uses the MLOps Stacks project that is in Public Preview. We encourage you to follow along as your infrastructure allows. If you're not able to complete all of the steps, we recommended following the demo closely. If you have access to Databricks Academy self-paced learning courses, there is a video walkthrough within the course content.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC * Cut a release branch for deploying a machine learning project to production
# MAGIC * Describe how to include monitoring in an MLOps Stacks project
# MAGIC * Describe how to add and edit orchestration workflows for an MLOps project
# MAGIC * Describe how to change compute resource configurations for an MLOps project

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * An MLOps Stacks project in a Git repo
# MAGIC
# MAGIC **Note**: If you do not have an existing MLOps Stacks project, we recommend that you set one up following the instructions [here]($../3.1 - Infrastructure Setup/3.1.1 - Infrastructure Setup Demo). If you do not have the appropriate instructions, you can at least review the code structure of the project in the "Production – Example Project" folder.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Cut a Release Branch
# MAGIC
# MAGIC The first step we have in deploying to production is cutting a `release` branch for our project repository.
# MAGIC
# MAGIC There are two ways to do this:
# MAGIC
# MAGIC 1. Create a new branch based off of the `main` branch
# MAGIC 2. Merge the latest from the `main` branch into an existing `release` branch
# MAGIC
# MAGIC For this demo, we'll create a new release branch since it doesn't yet exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy Solution
# MAGIC
# MAGIC When the release branch is created or has new code pushed to it, it will trigger a CD workflow to deploy the solution to the production environment.
# MAGIC
# MAGIC This deployment includes:
# MAGIC
# MAGIC 1. Project code
# MAGIC 2. Orchestration workflows
# MAGIC 3. Compute resources
# MAGIC
# MAGIC We can take a look at each one of these in our Databricks workspace.
# MAGIC
# MAGIC **Note**: This deployment was also in the development and staging workspace(s). We're just only now mentioning the specifics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Considerations
# MAGIC
# MAGIC There are many other questions related to deploying to production that we need to consider, including:
# MAGIC
# MAGIC * How do I use my own data?
# MAGIC * How do I monitor my solution?
# MAGIC * How do schedule my job refreshes?
# MAGIC * How do I change my compute resource configurations?
# MAGIC
# MAGIC We'll cover those now!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data
# MAGIC
# MAGIC Most of the data is configured in the `mlops_demo/resources` folder.
# MAGIC
# MAGIC Check out the files in that location to update the data used for the project.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Configuration
# MAGIC
# MAGIC Your compute resources are set in your `mlops_demo/resources` folder, as well.
# MAGIC
# MAGIC It contains information on:
# MAGIC
# MAGIC * Runtime
# MAGIC * Number of Workers
# MAGIC * Worker Node Type

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitoring
# MAGIC
# MAGIC As mentioned, MLOps Stacks is in Public Preview. At this time, it doesn't have any built-in monitoring solution – but there is a placeholder for it!
# MAGIC
# MAGIC The monitoring solution will utilize [Lakehouse Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html#introduction-to-databricks-lakehouse-monitoring), a tool that lets you monitor the statistical properties and quality of your data tables.
# MAGIC
# MAGIC This will help answer questions about drift and performance, including:
# MAGIC
# MAGIC * How are ML model inputs and predictions shifting over time?
# MAGIC * How is model performance trending over time? Is model version A performing better than version B?
# MAGIC
# MAGIC Demonstration will be provided as the MLOps Stacks project integrates its monitoring solution.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retraining
# MAGIC
# MAGIC So with monitoring coming, how do we know when to retrain our job?
# MAGIC
# MAGIC At this point, this comes down to scheduling timely refreshes of your deployed job runs.
# MAGIC
# MAGIC You'll want to run the following:
# MAGIC
# MAGIC * Feature preparation jobs
# MAGIC * Model training and validation jobs
# MAGIC * Deployment jobs
# MAGIC
# MAGIC #### Champion vs. Challenger
# MAGIC
# MAGIC MLOps Stacks has minimal Champion vs. Challenger logic built into its solution.
# MAGIC
# MAGIC The logic follows from training -> validation -> deployment -> inference.
# MAGIC
# MAGIC You can see the process in the following locations:
# MAGIC
# MAGIC 1. `mlops_demo/validation/notebooks/ModelValidation.py` - establishing a model as challenger model, and passing it to the next stage, if it passes validation checks (set up manually)
# MAGIC     * **Important**: to include comparing to champion performance as a validation check, update `enable_baseline_comparison` in `mlops_demo/resources/model-workflow-resource.yml` to `true`
# MAGIC 2. `mlops_demo/deployment/model_deployment/notebooks/ModelDeployment.py` - moves validation-passing model to deployment where it becomes the "champion" model
# MAGIC
# MAGIC **Note**: If you want to create more robust champion/challenger analysis, then focus on adding more validation steps to the ModelValidation notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Summary
# MAGIC
# MAGIC In this demonstration, you learned how to:
# MAGIC
# MAGIC * Cut a release branch for deploying a machine learning project to production
# MAGIC * Describe how to include monitoring in an MLOps Stacks project
# MAGIC * Describe how to add and edit orchestration workflows for an MLOps project
# MAGIC * Describe how to change compute resource configurations for an MLOps project

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>