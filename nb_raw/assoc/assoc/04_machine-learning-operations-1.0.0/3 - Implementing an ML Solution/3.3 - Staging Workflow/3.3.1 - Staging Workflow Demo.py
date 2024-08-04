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
# MAGIC # Staging Workflow Demo
# MAGIC
# MAGIC In this demo, we will provide a walkthrough of the staging workflow for machine learning projects according to MLOps best practices using MLOps Stacks.
# MAGIC
# MAGIC **Note**: This demonstration uses the MLOps Stacks project that is in Public Preview. We encourage you to follow along as your infrastructure allows. If you're not able to complete all of the steps, we recommended following the demo closely. If you have access to Databricks Academy self-paced learning courses, there is a video walkthrough within the course content.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC * Examine the tests and CI/CD workflows set up by MLOps Stacks
# MAGIC * Run tests by start a merge of development code into your `main` branch
# MAGIC * Deploy and run your MLOps Stacks project in your staging environment by approving a merge

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * An MLOps Stacks project in a Git repo
# MAGIC
# MAGIC **Note**: If you do not have an existing MLOps Stacks project, we recommend that you set one up following the instructions [here]($../3.1 - Infrastructure Setup/3.1.1 - Infrastructure Setup Demo). If you do not have the appropriate instructions, you can at least review the code structure of the project in the "Staging – Example Project" folder.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Review CI/CD Workflows and Tests
# MAGIC
# MAGIC The Staging stage of the MLOps recommended workflow is all about testing – we want to test that our workflows and project code are set up appropriately.
# MAGIC
# MAGIC Before we do that, we need to understand what tests to run.
# MAGIC
# MAGIC We'll begin this by examining the MLOps Stacks project to learn what's included in the CI/CD workflows and preconfigured tests.
# MAGIC
# MAGIC #### CI/CD Workflows
# MAGIC
# MAGIC Our CI/CD workflows are defined in our `<cicd-tooling>/workflows` subfolder of our project.
# MAGIC
# MAGIC For this demo, you will see files like:
# MAGIC
# MAGIC * `.github/workflows/lint-cicd-workflow-files.yml`
# MAGIC * `.github/workflows/mlops_demo-bundle-cd-staging.yml`
# MAGIC
# MAGIC These files define our CI/CD workflows for different scenarios.
# MAGIC
# MAGIC For example, `lint-cicd-workflow-files.yml` sets up a testing workflow to ensure our CI/CD workflows themselves are set up correctly.
# MAGIC
# MAGIC #### Tests
# MAGIC
# MAGIC So where are the tests actually located?
# MAGIC
# MAGIC Well, we can see a few in the `<cicd-tooling>/workflows` subfolder itself:
# MAGIC
# MAGIC * The `<project_name>-run-tests.yml` file shows what commands are run for unit tests and integration tests
# MAGIC
# MAGIC **Note**: You'll probably want to run more unit tests for your project! You can do this by adding runs for the notebooks in the `tests` project subfolder to your testing workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Pull Request to Run Tests
# MAGIC
# MAGIC When we're ready to run our tests, we can create a pull request to merge our `dev` branch into our `main`.
# MAGIC
# MAGIC This will run the following tests:
# MAGIC
# MAGIC 1. Check for branch conflicts
# MAGIC 2. Validate the CI/CD setup using Lint
# MAGIC 3. Validate the project using the `validate` tool including in the MLOps Stacks tooling on both the `staging` and `prod` environments
# MAGIC 4. Run the unit tests
# MAGIC 5. Run the integration tests
# MAGIC
# MAGIC You can view the status of the tests directly in your Git provider.
# MAGIC
# MAGIC **Note**: All of this will run and you will see the results prior to making the decision to merge your code. This ensures that you're only merging working code into your `main` branch.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Merge Pull Request and Deploy to Staging
# MAGIC
# MAGIC When our project code passes its tests, we'll be ready to merge our code into the `main` branch.
# MAGIC
# MAGIC This will trigger one more test: a project deployment and run within the `staging` environment.
# MAGIC
# MAGIC **Note**: You will need to head to the Actions tab in GitHub to view the results of these tests. Before moving to production, you should ensure that this deployment was successful.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Summary
# MAGIC
# MAGIC In this demonstration, you learned how to:
# MAGIC
# MAGIC * Examine the tests and CI/CD workflows set up by MLOps Stacks
# MAGIC * Run tests by start a merge of development code into your `main` branch
# MAGIC * Deploy and run your MLOps Stacks project in your staging environment by approving a merge

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>