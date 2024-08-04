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
# MAGIC # Development Workflow Demo
# MAGIC
# MAGIC In this demo, we will provide a walkthrough of the development workflow for machine learning projects according to MLOps best practices using MLOps Stacks.
# MAGIC
# MAGIC **Note**: This demonstration uses the MLOps Stacks project that is in Public Preview. We encourage you to follow along as your infrastructure allows. If you're not able to complete all of the steps, we recommended following the demo closely. If you have access to Databricks Academy self-paced learning courses, there is a video walkthrough within the course content.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC * Make changes to MLOps Stacks pipeline code specific to an ML project
# MAGIC * Validate that your MLOps Stacks project is configured correctly
# MAGIC * Deploy your MLOps Stacks project in your development environment
# MAGIC * Tear down your MLOps Stacks project from your development environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * An MLOps Stacks project in a Git repo
# MAGIC
# MAGIC **Note**: If you do not have an existing MLOps Stacks project, we recommend that you set one up following the instructions [here]($../3.1 - Infrastructure Setup/3.1.1 - Infrastructure Setup Demo). If you do not have the appropriate infrastructure, you can at least review the code structure of the project in the "Dev – Example Project" folder.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Make Changes to Code
# MAGIC
# MAGIC When you're starting a machine learning project using MLOps Stacks, you're going to need to update the project code in some way.
# MAGIC
# MAGIC This could include:
# MAGIC
# MAGIC * Updated environment configuration
# MAGIC * Updated modeling techniques
# MAGIC * New model validation criteria
# MAGIC * etc.
# MAGIC
# MAGIC These changes can be made in the following locations of the MLOps Stacks project:
# MAGIC
# MAGIC * requirements.txt
# MAGIC * Train notebook
# MAGIC * Validation notebook
# MAGIC * etc.
# MAGIC
# MAGIC **Note**: Whenever you make changes, note that they'll be picked up by the preconfigured jobs via Databricks Workflows. If you'd like to change the Jobs that are created, you can do so in the **assets** folder of the project.
# MAGIC
# MAGIC Whatever you're doing, it's important to follow the below general workflow:
# MAGIC
# MAGIC 1. Ensure you've checked out your `dev` branch
# MAGIC 2. Make updates to the project code
# MAGIC 3. *Optional*: Commit your project code
# MAGIC
# MAGIC **Note**: Step 3 above is Optional if you are using a local IDE, but it's required when working with a cloud-based IDE like Databricks.
# MAGIC
# MAGIC Feel free to follow along to the demonstration of this approach in Databricks Academy or in an instructor-led delivery of this course.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Validate Code
# MAGIC
# MAGIC When you're working within a predefined project structure like MLOps Stacks has, it's important to make sure your changes don't break the project's automated capabilities.
# MAGIC
# MAGIC MLOps Stacks makes this easy by providing the ability to validate your project code.
# MAGIC
# MAGIC You can do this by running the following:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC databricks bundle validate
# MAGIC ```
# MAGIC
# MAGIC **Note**: A successful validation will return a JSON of the bundle project.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Deploy and Run Solution
# MAGIC
# MAGIC After we've validated our code, we'll want to **Deploy** and **Run** our project.
# MAGIC
# MAGIC * When we **Deploy** our project, we are building our project within a Databricks workspace – this will create the necessary project structure, compute resources, and orchestration workflows within the workspace.
# MAGIC * When we **Run** our project, we are running the orchcestration workflows within a Databricks workspace.
# MAGIC
# MAGIC It's important to first **Deploy** and then **Run**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Project
# MAGIC
# MAGIC To deploy our project, we can run the following from our command line environment:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC databricks bundle deploy -t <target-name>
# MAGIC ```
# MAGIC
# MAGIC Where `<target-name>` is the name of the desired target within the `databricks.yml` file, for example `dev`, `staging`, or `prod`. In our case, we'll use `dev` during this workflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the Project
# MAGIC
# MAGIC Once the project is deployed, we can run the project workflows to verify that they run in our development environment.
# MAGIC
# MAGIC While the jobs will eventually run on their own based on the schedule defined in MLOps Stacks, let's run them immediately with one of the two below options:
# MAGIC
# MAGIC * Interactively from the workspace (demonstrated here)
# MAGIC * From the command line using `databricks bundle run -t <target-name> <job-name>`
# MAGIC
# MAGIC **Note**: `<job-name>` can be found in the **Resources** subfolder files.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Tear Down Solution
# MAGIC
# MAGIC After the project is validated and is confirmed that it can run in the development environment, it's smart to tear down the solution from the development environment. 
# MAGIC
# MAGIC This helps keep the development environment clean and free of clutter. And since the project is comprehensively represented in the MLOps Stacks project repository, we can always rebuild it later when needed.
# MAGIC
# MAGIC To destroy the project deployment, run the following:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC databricks bundle destroy -t <target-name>
# MAGIC ```
# MAGIC
# MAGIC **Note**: You might be prompted to delete other assets (like registered models) first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Summary
# MAGIC
# MAGIC In this demonstration, you learned how to:
# MAGIC
# MAGIC * Make changes to MLOps Stacks pipeline code specific to an ML project
# MAGIC * Validate that your MLOps Stacks project is configured correctly
# MAGIC * Deploy your MLOps Stacks project in your development environment
# MAGIC * Tear down your MLOps Stacks project from your development environment

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>