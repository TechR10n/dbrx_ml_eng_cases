# Databricks notebook source

import re

username = (sc._jvm.com.databricks.logging.AttributionContext.current().tags().get(
  sc._jvm.com.databricks.logging.BaseTagDefinitions.TAG_USER()).x())

userhome = f"dbfs:/user/{username}"

dbutils.fs.mkdirs(userhome + "/tmp")

database = f"""{re.sub("[^a-zA-Z0-9]", "_", username)}_var_db"""

dbutils.widgets.text("mode", "setup")
mode = dbutils.widgets.get("mode")

if mode == "reset":
  spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")
  spark.sql(f"DROP DATABASE IF EXISTS {database}_dev CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

spark.sql(f"USE {database}");
