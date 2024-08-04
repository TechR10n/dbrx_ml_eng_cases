# Databricks notebook source
# version = spark.conf.get("dbacademy.library.version", "v2.0.5")

# if not version.startswith("v"): library_url = f"git+https://github.com/databricks-academy/dbacademy@{version}"
# else: library_url = f"https://github.com/databricks-academy/dbacademy/releases/download/{version}/dbacademy-{version[1:]}-py3-none-any.whl"

library_url = "/dbfs/FileStore/whl/dbacademy-2.0.5-py3-none-any.whl"


default_command = f"install --quiet --disable-pip-version-check {library_url}"
pip_command = spark.conf.get("dbacademy.library.install", default_command)

if pip_command != default_command:
    print(f"WARNING: Using alternative library installation:\n| default: %pip {default_command}\n| current: %pip {pip_command}")

# COMMAND ----------

# MAGIC %pip $pip_command