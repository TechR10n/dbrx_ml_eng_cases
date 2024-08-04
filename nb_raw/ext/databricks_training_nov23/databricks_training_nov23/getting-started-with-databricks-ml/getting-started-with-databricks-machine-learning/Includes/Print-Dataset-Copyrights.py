# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

lesson_config.create_schema = False                 # We don't need a schema when simply printing the copyrights

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.print_copyrights()                               # Once initialized, just print the copyrights

