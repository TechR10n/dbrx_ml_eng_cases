# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Simulate Multiple Trainings
# MAGIC
# MAGIC In this notebook, you'll refactor the code from previous notebooks to create a single notebook that can be scheduled as a job. At the end of this notebook, your instructor will teach you how to execute notebooks as jobs.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the of this lesson, students should be able to:
# MAGIC - Explain how shallow clones enable easy creation of development tables
# MAGIC - Use widgets to parameterize notebooks
# MAGIC - Perform an insert-only merge into a Delta Lake table
# MAGIC - Use MLflow to track a notebook-based experiment
# MAGIC - Describe how MLflow, Delta Lake, Spark, and open-source Python packages combine to provide end-to-end ML pipeline management

# COMMAND ----------

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.window import Window
import mlflow
import statsmodels.api as sm
import pyspark
from datetime import datetime

portfolio_table = 'portfolio'
stock_table = 'stock'
stock_return_table = 'stock_return'
market_table = 'market'
market_return_table = 'market_return'
trial_table = 'monte_carlo'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set MLflow Experiment
# MAGIC
# MAGIC When using MLflow with a notebook scheduled as a job, you'll need to set the experiment using the **path of the notebook**.

# COMMAND ----------

# TODO
try:
  mlflow.set_experiment("<FILL-IN>")
except:
  print("Copy the notebook path from the UI and replace the <FILL-IN> above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Widgets to Pass Parameters
# MAGIC
# MAGIC Databricks notebook [widgets](https://docs.databricks.com/notebooks/widgets.html) enable parameterization of notebooks when [running jobs](https://docs.databricks.com/dev-tools/api/latest/jobs.html#parampair).
# MAGIC
# MAGIC The cell below creates and captures the value for a widget with the key `run_date`. You'll add a couple of additional parameters as you progress through this notebook.

# COMMAND ----------

dbutils.widgets.text("run_date", "yyyy-MM-dd")
run_date = dbutils.widgets.get("run_date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refactor Setup Script
# MAGIC
# MAGIC Throughout the course, we've been running a notebook [`./Includes/setup`]($./Includes/setup) to configure a number of variables that are being used for reproducibility. This code is copied into the cell below.
# MAGIC
# MAGIC | variable | explanation |
# MAGIC | --- | --- |
# MAGIC | `username` | This is the user's sign-in address, captured from internal tags used for logging. This variable is used exclusively for formatting string values for the `userhome` and `database` variables.|
# MAGIC | `userhome` | A unique path within the DBFS root so that students do not conflict when running this demo in the same workspace. |
# MAGIC | `database` | A unique database name; this is used to declare a database that should not conflict with other databases in the user workspace. |
# MAGIC
# MAGIC Because of differences in execution environment between jobs clusters and all-purpose clusters, these internal tags should not be relied upon to capture the username. Replace this logic by using a widget with the key `username`. Make sure you capture this value back to a Python variable `username`.

# COMMAND ----------

# TODO

import re

username = (sc._jvm.com.databricks.logging.AttributionContext.current().tags().get(
  sc._jvm.com.databricks.logging.BaseTagDefinitions.TAG_USER()).x())

userhome = f"dbfs:/user/{username}"

database = f"""{re.sub("[^a-zA-Z0-9]", "_", username)}_var_db"""

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

spark.sql(f"USE {database}");

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Shallow Clone to Create Dev Tables
# MAGIC
# MAGIC Delta Lake allows no-copy clones to be created using the [shallow clone](https://docs.databricks.com/delta/delta-utility.html#clone-delta-table) SQL phrase. This provides an easy an efficient method to test development code against a fork of production data without any need to duplicate data.
# MAGIC
# MAGIC The cell below defines logic to capture every table name from the current database and create a clone in your `_dev` database if that table doesn't exist. Add logic to use a widget with the keyword `env` that allows you to run this code if the value `DEV` is passed. Optionally, add logic to delete all your cloned tables and create new clones.

# COMMAND ----------

# TODO
database_dev = database + "_dev"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_dev}")

spark.sql(f"USE {database_dev}");

for table in spark.catalog.listTables(database):
  spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {database_dev}.{table[0]}
  SHALLOW CLONE {database}.{table[0]}
  """
  )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Log Table Versions
# MAGIC
# MAGIC Prior to beginning our run, we'll log the versions of each of our Delta Lake tables. We'll log table versions again at the end of the notebook run, making it easy to track changes that may have occurred during the run.
# MAGIC
# MAGIC In the cell below, define a reusable function to log the version of a Delta Lake table as an MLflow parameter. To track the changing versions of our tables during the run, build in logic to pass an additional integer value `v` to record what point in the run you're logging the table. **Make sure your table has control logic to avoid failing on temp views**.
# MAGIC
# MAGIC To name your logged parameters, use the format `v<v>.<table_name>`. Below are examples of how you can capture the table version.
# MAGIC
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> While Delta Lake provides ACID guarantees on your tables, multiple jobs can be modifying a table at once without conflict. In a production setting, you may wish to log data versions in the functions you use to train your models to prevent versioning errors.

# COMMAND ----------

# TODO

delta_m_version = sql("DESCRIBE HISTORY " + market_return_table).select("version").collect()[0][0]
delta_s_version = sql("DESCRIBE HISTORY " + stock_return_table).select("version").collect()[0][0]

def logTableVersion(tablename:str, v:int):
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC Now use your function to log the version of all tables in your database.

# COMMAND ----------

for table in spark.catalog.listTables():
  logTableVersion(table.name, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Environmental Variables
# MAGIC
# MAGIC To differentiate our runs, we'll log our three widgets as parameters.

# COMMAND ----------

mlflow.log_param("env", env)
mlflow.log_param("run_date", run_date)
mlflow.log_param("username", username)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Append New Stock Data
# MAGIC
# MAGIC You'll need to make sure to specify the `yfinance` library dependency when deploying to a jobs cluster. 
# MAGIC
# MAGIC While designing our pipeline, we'll build some assumptions into our logic to simulate real world requirements:
# MAGIC 1. Once committed, we do not wish to drop or re-write ingested data.
# MAGIC 1. We may encounter data arriving up to 30 days late; we should ingest these records correctly and fill in our table.
# MAGIC 1. We wish to download the minimal amount of data necessary to check for potential late-arriving records.
# MAGIC
# MAGIC Much of this will be handled in how we write our data to Delta Lake. We will need to adapt our `fetch_tick` function to vary the start and end date.
# MAGIC
# MAGIC When using [`applyInPandas`](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#grouped-map) to pass data efficiently from Spark DataFrames to Pandas and the Python environment, the first positional argument for the defined function will always be the keys that the `groupBy` is called against. In this instance, we can pass arbitrary parameters within our `groupBy` using `F.lit`. 
# MAGIC
# MAGIC Update the code below to capture a `start_date` and `end_date` argument and pass these on to `yf.download`.

# COMMAND ----------

# TODO

import yfinance as yf

schema = """
  ticker string, 
  date date,
  open double,
  high double,
  low double,
  close double,
  volume double
"""

def fetch_tick(group, pdf):
  tick = group[0]
  # <FILL_IN>
  try:
    raw = yf.download(tick, start="2017-01-01", end="2020-02-07")[['Open', 'High', 'Low', 'Close', 'Volume']]
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    output_df = raw.reindex(idx, method='pad')
    # Pandas does not keep index (date) when converted into spark dataframe
    output_df['date'] = output_df.index
    output_df['ticker'] = tick    
    output_df = output_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Volume": "volume", "Close": "close"})
    return output_df
  except:
    return pd.DataFrame(columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])

# COMMAND ----------

# MAGIC %md
# MAGIC Use the `run_date` that will be provided via a widget as your `end_date`. Write a Spark SQL function to set the `start_date` variable. Make sure both variables are string with the format `yyyy-MM-dd`.

# COMMAND ----------

# TODO

end_date = run_date
# start_date = spark.sql(f"<FILL-IN>")

# COMMAND ----------

# MAGIC %md
# MAGIC Delta Lake supports an optimized [insert-only merge](https://docs.databricks.com/delta/delta-update.html#data-deduplication-when-writing-into-delta-tables) to automatically avoid appending duplicate records while loading new data into existing tables.
# MAGIC
# MAGIC Update the code below to:
# MAGIC 1. Use `start_date` and `end_date` in the `groupBy` clause
# MAGIC 1. Register the DataFrame resulting from `applyInPandas` as a temporary view
# MAGIC 1. Use a SQL statement to perform a insert-only merge into your stock table.

# COMMAND ----------

# TODO

(spark
  .read
  .table(portfolio_table)
  .groupBy("ticker")
  .applyInPandas(fetch_tick, schema=schema)
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(stock_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Market Factors
# MAGIC
# MAGIC We'll perform similar refactoring for the code we used to download our market factors; transform the code below to use our `start_date` and `end_date` variables, and then append the new records using an insert-only merge.

# COMMAND ----------

# TODO
factors = {
  '^GSPC':'SP500',
  '^NYA':'NYSE',
  '^XOI':'OIL',
  '^TNX':'TREASURY',
  '^DJI':'DOWJONES'
}

# Create a pandas dataframe where each column contain close index
factors_df = pd.DataFrame()
for tick in factors.keys():    
    raw = yf.download(tick, start="2017-01-01", end="2020-02-07")
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    pdf = raw.reindex(idx, method='pad')
    factors_df[factors[tick]] = pdf['Close'].copy()
        
# Pandas does not keep index (date) when converted into spark dataframe
factors_df['Date'] = idx

# (spark.createDataFrame(factors_df)
#   .write
#   .format("delta")
#   .mode("overwrite")
#   .saveAsTable(market_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Market and Stock Return Tables
# MAGIC
# MAGIC Here we'll reuse our code and `overwrite` our Delta Lake tables each time the notebook is run. Recording the version of the Delta table associated with each run provides the means to fully recreate an experiment. For long-term versioning requirements, changes to the [default retention settings](https://docs.databricks.com/delta/delta-batch.html#data-retention) are required. (Note that for Delta Lake's [deep clone](https://docs.databricks.com/spark/latest/spark-sql/language-manual/clone.html) functionality is prescribed for retaining historical table snapshots.)

# COMMAND ----------

# our market factors easily fit in memory, use pandas for convenience
df = spark.table(market_table).toPandas()

# add date column as pandas index for sliding window
df.index = df['Date']
df = df.drop(columns = ['Date'])

# compute daily log returns
df = np.log(df.shift(1)/df)

# add date columns
df['date'] = df.index

# overwrite log returns to market return table
(spark.createDataFrame(df)
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(market_return_table))

# COMMAND ----------

# Apply a tumbling 1 day window on each instrument
window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)

# apply sliding window and take first element
# compute returns
# make sure we have corresponding dates in market factor tables
sdf = (spark.table(stock_table)
  .filter(F.col('close').isNotNull())
  .withColumn("first", F.first('close').over(window))
  .withColumn("return", F.log(F.col('close')/F.col('first')))
  .select('date', 'ticker', 'return')
  .join(spark.table(market_return_table), 'date')
  .select('date', 'ticker', 'return'))

# overwrite log returns to stock return table
(sdf.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(stock_return_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train OLS Model with Updated Data
# MAGIC
# MAGIC Our chosen modeling approach uses our full historical market data to make predictions for each stock's daily P&L. As such, prior to making our new VAR estimates, we'll need to update the parameters of our models.
# MAGIC
# MAGIC We'll use the same basic code that we used earlier, with one major change: instead of outputting our model parameters to JSON, we'll add `run_date` and `run_uuid` columns to our DataFrame and save them as a Delta Lake table.
# MAGIC
# MAGIC We'll start by defining an empty Delta Lake with the desired schema.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS model_params
# MAGIC (run_date DATE, run_uuid STRING, ticker STRING, weights ARRAY<FLOAT>)
# MAGIC USING DELTA

# COMMAND ----------

# MAGIC %md
# MAGIC Our `run_uuid` column will allow us to ensure that we're using the correct rows of our Delta Table, even if we decide to run multiple versions of the model on the same date.

# COMMAND ----------

run_uuid = mlflow.active_run().info.run_uuid

# COMMAND ----------

# MAGIC %md
# MAGIC Now we'll update our code add these new columns and append to a Delta Table (instead of casting our data back to Pandas to save as a JSON).

# COMMAND ----------

# TODO

feature_names = ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES']

x_train = (spark.table(market_return_table)
  .withColumn("features", F.array(feature_names))
  .dropna()
  .select('date', 'features')
  .join(spark.table(stock_return_table), 'date'))

# add non linear transformations as simple example on non linear returns
def featurize(xs):
  fs = []
  for x in xs:
    fs.append(x)
    fs.append(np.sign(x) * x**2)
    fs.append(x**3)
    fs.append(np.sign(x) * np.sqrt(abs(x)))
  return fs
  
# use pandas UDF to train multiple model (one for each instrument) in parallel
# the resulting dataframe will be the linear regression weights for each instrument
schema = "ticker string, weights array<float>"

def train_model(group, pdf):
  # build market factor vectors
  # add a constant - the intercept term for each instrument i.
  X = [featurize(row) for row in np.array(pdf['features'])]
  X = sm.add_constant(X, prepend=True) 
  y = np.array(pdf['return'])
  model = sm.OLS(y, X).fit()
  w_df = pd.DataFrame(data=[[model.params]], columns=['weights'])
  w_df['ticker'] = group[0]
  return w_df

# the resulting dataframe easily fits in memory and will be saved as our "uber model", serialized to json
models_df = x_train.groupBy('ticker').applyInPandas(train_model, schema=schema).toPandas()
# models_path = f"{userhome.replace("dbfs:", "/dbfs")}/tmp/models.json"
# models_df.to_json(models_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refactor Parameter Loading for Predictions
# MAGIC
# MAGIC Note that the prediction code copied below is written in pure Python and re-uses the output of the `featurize` method from above. This code is written to accept two arrays:
# MAGIC
# MAGIC | variable | description |
# MAGIC | --- | --- |
# MAGIC | `fs` | The market features used to make our prediction (transformed by `featurize`) |
# MAGIC | `ps` | The parameters learned from OLS training for our current stock |
# MAGIC
# MAGIC Note that our function is presently written to perform a single prediction with each execution.

# COMMAND ----------

def predict(fs, ps):
  s = ps[0]
  for i, f in enumerate(fs):
    s = s + ps[i + 1] * f
  return float(s)

# COMMAND ----------

# MAGIC %md
# MAGIC The code below was used to reload the previous models saved as JSON. We broadcast the models to the executors, so that we can parallelize our Monte Carlo simulation.
# MAGIC
# MAGIC Refactor the code below to define the `loadModels` function, making sure you filter by `run_uuid` and correctly broadcast each model as a key:value pair for `ticker`:`weights`.

# COMMAND ----------

# TODO

models = {}
for model in np.array(pd.read_json(model_path)):
  models[model[0]] = model[1]

model_dict = spark.sparkContext.broadcast(models)

def loadModels(table_name:str, run_uuid:str) -> pyspark.broadcast.Broadcast:
  # <FILL-IN>
  return model_dict

model_dict = loadModels("model_params", run_uuid)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Monte Carlo Simulation
# MAGIC
# MAGIC Now that we have an updated OLS model, we'll re-run our Monte Carlo simulation.
# MAGIC
# MAGIC To simplify our code, we'll refactor our `compute_volatility` function to:
# MAGIC 1. Accept 1 argument, `table_name`, as a string
# MAGIC 1. Load 90 days of data from our `market_returns` table
# MAGIC 1. Return the mean and covariance of each field

# COMMAND ----------

# TODO
  
def compute_volatility(f_ret_pdf):
  # Retrieve market factor covariance matrix and average of returns
  # This will be used to generate a multi variate distribution of market factor returns
  return np.array(f_ret_pdf.mean()), np.array(f_ret_pdf.cov())

# COMMAND ----------

# MAGIC %md
# MAGIC Reuse the code for the `simulate_market` and `run_trials` functions.

# COMMAND ----------

def simulate_market(f_ret_avg_pdf, f_ret_cov_pdf, seed):
  np.random.seed(seed = seed)
  return np.random.multivariate_normal(f_ret_avg_pdf, f_ret_cov_pdf)

def run_trials(pdf):
  
  # Deserialize objects from cache
  models = model_dict.value
  f_ret_avg = f_ret_avg_B.value
  f_ret_cov = f_ret_cov_B.value
  
  trials = []
  for seed in np.array(pdf.seed):
    market_condition = simulate_market(f_ret_avg, f_ret_cov, seed)
    market_features = featurize(market_condition)
    for ticker in models.keys(): 
      trial = predict(market_features, models[ticker])
      trials.append([ticker, seed, trial])
    
  # Return a dataframe with each simulation across instruments per row
  trials_pdf = pd.DataFrame(data=trials, columns=['ticker', 'seed', 'trial'])
  return trials_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Trial Table Schema
# MAGIC
# MAGIC To increase our reproducibility, you've decided to add the `run_uuid` to the trial table.
# MAGIC
# MAGIC Luckily, Delta Table allows us to easily [update our table schema](https://docs.databricks.com/spark/latest/spark-sql/language-manual/alter-table-or-view.html#delta-lake-schema-constructs) without rewriting our data.
# MAGIC
# MAGIC In the cell below, write a simple control flow statement that checks if `run_uuid` is in the trial table, and if not, add this column after `run_date`.

# COMMAND ----------

# TODO

spark.table(trial_table).columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update Monte Carlo Simulation Code
# MAGIC
# MAGIC Below is the code used previously to run our Monte Carlo simulations. We've updated several of our functions, arguments, and tables, and will need to update our code to reflect these changes.
# MAGIC
# MAGIC 1. `to_date` and `from_date` are no longer necessary, as we've factored out our `retrieve_market_factors` function. Make sure any references for `trial_date` or `to_date` are changed to use the `run_date` variable
# MAGIC 1. The code to define `seed_init` needs to be updated to use `run_date`; make sure you appropriately cast this string value to unixtime
# MAGIC 1. `compute_volatility` now takes the arguments `table_name` and `run_date`
# MAGIC 1. The `trial_table` needs to have the `run_uuid` inserted between `run_date` and `ticker`

# COMMAND ----------

# TODO

# how many simulations do we want to run (industry standard ~ 20,000)
runs = 50000

# how many executors can run in parallel
parallelism = 8

# Control experiment
to_date = trial_date.strftime("%Y-%m-%d")
from_date = (trial_date - timedelta(days = d_days)).strftime("%Y-%m-%d")
seed_init = int(trial_date.timestamp())

# create a dataframe of seeds so that each trial will result in a different simulation
# each executor is responsible for num_instruments * ( total_runs / num_executors ) trials
seed_pdf = pd.DataFrame([[seed_init + x, x % parallelism] for x in np.arange(0, runs)], columns = ['seed', 'executor'])
seed_df = spark.createDataFrame(seed_pdf).repartition(parallelism, 'executor')
seed_df.cache()
seed_df.count()

# Compute volatility
market_df = retrieve_market_factors(from_date, to_date)
f_ret_avg, f_ret_cov = compute_volatility(market_df)
f_ret_avg_B = spark.sparkContext.broadcast(f_ret_avg)
f_ret_cov_B = spark.sparkContext.broadcast(f_ret_cov)

# group dataframe of seeds at the executor level and run simulations
mc_df = seed_df.groupBy('executor').applyInPandas(run_trials, schema = 'ticker string, seed int, trial float')

# store runs
(mc_df
  .withColumn('run_date', F.lit(to_date))
  .join(spark.read.table(portfolio_table), 'ticker', 'inner')
  .select('run_date', 'ticker', 'seed', 'trial', 'industry', 'country')
  .write
  .partitionBy("run_date")
  .mode("append")
  .format("delta")
  .saveAsTable(trial_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Table Versions
# MAGIC
# MAGIC Reuse your `logTableVersion` function defined earlier in the notebook. Pass the integer `1` for your argument `v`.

# COMMAND ----------

for table in spark.catalog.listTables():
  logTableVersion(table.name, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## End MLflow Run
# MAGIC
# MAGIC Now that all your parameters have been saved, you can end the run to commit these parameters.

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Run this Notebook as a Job
# MAGIC
# MAGIC Your instructor will now walk you through using the [Jobs UI](https://docs.databricks.com/jobs.html) to create and run a job with parameters. They will then walk you through submitting jobs with the [Jobs CLI](https://docs.databricks.com/dev-tools/cli/jobs-cli.html); in order to submit and manage jobs from the command line, you'll need to install and configure the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html).
# MAGIC
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> Your instructor may choose to use an existing all purpose cluster when demonstrating these tasks. In production, you should use jobs clusters to reduce operating expenses.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>