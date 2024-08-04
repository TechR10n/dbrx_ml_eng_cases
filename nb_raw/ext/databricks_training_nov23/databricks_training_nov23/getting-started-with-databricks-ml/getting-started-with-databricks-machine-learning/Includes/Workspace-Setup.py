# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Init Script & Install Datasets
# MAGIC The main affect of this call is to pre-install the datasets.
# MAGIC
# MAGIC It has the side effect of create our DA object which includes our REST client.

# COMMAND ----------

lesson_config.create_schema = False                 # We don't need a schema when configuring the workspace

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student
