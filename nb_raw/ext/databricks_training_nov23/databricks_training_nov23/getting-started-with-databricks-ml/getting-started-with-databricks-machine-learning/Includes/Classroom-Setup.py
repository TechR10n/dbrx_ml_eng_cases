# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
#DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

dbutils.fs.mkdirs(DA.paths.working_dir)

# This has been replaced by DA.paths.datasets...
base_read_path = "wasbs://courseware@dbacademy.blob.core.windows.net/introduction-to-databricks-machine-learning/v01/"

DA.paths.retail_cust_read_path = f"{DA.paths.datasets}/retail-org/customers"

# Replace this with that DA.paths.working_dir
# base_write_path = DA.paths.working_dir

DA.paths.input_path = f"{DA.paths.retail_cust_read_path}/customers.csv"

# Allows us to test MLflow jobs when under test.
DA.init_mlflow_as_job()

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student
