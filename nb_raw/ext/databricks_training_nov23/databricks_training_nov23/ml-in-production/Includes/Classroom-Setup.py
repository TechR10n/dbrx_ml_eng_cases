# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# Variable to catch bugs introduced by the rename of the notebook
real_time_lab = dbgems.get_notebook_name() == "02-Real-Time-Lab"

if real_time_lab:
    # Fixes a concurrency error with running 02-Real-Time and 02-Real-Time-Lab
    # where the setup & cleanup delete the same set of model serving endpoints.
    lesson_config.name = "L"

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)
DA.reset_lesson()
DA.init()

DA.paths.datasets_path = DA.paths.datasets.replace("dbfs:/", "/dbfs/")
DA.paths.working_path = DA.paths.working_dir.replace("dbfs:/", "/dbfs/")

if dbgems.get_notebook_path().endswith("/01-Experimentation/01-Feature-Store"):
    DA.paths.airbnb = f"{DA.paths.working_dir}/airbnb/airbnb.delta"

DA.init_mlflow_as_job()

DA.conclude_setup()
