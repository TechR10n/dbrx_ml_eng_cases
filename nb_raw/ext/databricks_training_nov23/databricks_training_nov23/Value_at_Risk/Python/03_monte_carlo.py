# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Monte Carlo Simulation
# MAGIC
# MAGIC In this notebook, we use our 40 models created in previous stage and runs million of monte carlo simulation in parallel using **Apache Spark**. For each simulated market condition sampled from a multi variate distribution, we predict our hypothetical instrument returns. By storing all of our data back into **Delta Lake**, we create a data asset that can be queried on-demand (as opposition to end of day) across multiple down stream use cases

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Control parameters
# MAGIC
# MAGIC In addition to our setup script, imports, and table name variable declarations, we'll define a handful of input parameters to use later in the notebook.

# COMMAND ----------

# MAGIC %run "./Includes/setup"

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from pyspark.sql import functions as F

# COMMAND ----------

portfolio_table = 'portfolio'
stock_table = 'stock'
stock_return_table = 'stock_return'
market_table = 'market'
market_return_table = 'market_return'
trial_table = 'monte_carlo'

# when do we want to simulate data
trial_date = datetime.strptime('2020-02-07', '%Y-%m-%d')

# where did we log our model
# model_path = 'ml:/var/production'
model_path = f"{userhome.replace('dbfs:', '/dbfs')}/tmp/models.json"

# how much history do we want compute volatility from
d_days = 90

# how many simulations do we want to run (industry standard ~ 20,000)
runs = 50000

# how many executors can run in parallel
parallelism = 8

# our predictive market factors
feature_names = ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES']

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Retrieve models and data
# MAGIC
# MAGIC We'll load our linear regression coefficients as `{ticker:weights}` key-value pairs. Broadcast this variable makes them availabe on each executor in the cluster.

# COMMAND ----------

# models we serialized as json from pandas dataframe
# we load models as dictionary of instrument <-> weights
models = {}
for model in np.array(pd.read_json(model_path)):
  models[model[0]] = model[1]
  
model_dict = spark.sparkContext.broadcast(models)

# COMMAND ----------

# MAGIC %md
# MAGIC Our input parameters provided above are used to sample 90 days of data from our market features.

# COMMAND ----------

def retrieve_market_factors(from_date, to_date):
  
  # Retrieve market factor returns in the provided time window
  from_ts = F.to_date(F.lit(from_date)).cast(TimestampType())
  to_ts = F.to_date(F.lit(to_date)).cast(TimestampType())
  f_ret = (spark.table(market_return_table)
    .filter(F.col('date') > from_ts)
    .filter(F.col('date') <= to_ts)
    .orderBy(F.asc('date'))
    .dropna())

  # Market factors easily fit in memory and will be used to create multivariate distribution of normal returns
  f_ret_pdf = f_ret.toPandas()
  f_ret_pdf.index = f_ret_pdf['date']
  f_ret_pdf = f_ret_pdf.drop(['date'], axis=1)
  return f_ret_pdf

# SAMPLE DATA
to_date = (datetime.now()).strftime("%Y-%m-%d")
from_date = (datetime.now() - timedelta(days = d_days)).strftime("%Y-%m-%d")
market_factors_df = retrieve_market_factors(from_date, to_date)
market_factors_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Generate market conditions
# MAGIC
# MAGIC We'll sample from a multivariate normal distribution. We'll begin by calculating the mean and covariance for our market factors.

# COMMAND ----------

def compute_volatility(f_ret_pdf):
  # Retrieve market factor covariance matrix and average of returns
  # This will be used to generate a multi variate distribution of market factor returns
  return np.array(f_ret_pdf.mean()), np.array(f_ret_pdf.cov())

f_ret_avg, f_ret_cov = compute_volatility(market_factors_df)
f_ret_avg

# COMMAND ----------

# MAGIC %md
# MAGIC Now we'll define methods to allow us to simulate potential market outcomes.

# COMMAND ----------

# generate same feature vectors as used at model training phase
# add non linear transformations as simple example on non linear returns
def featurize(xs):
  fs = []
  for x in xs:
    fs.append(x)
    fs.append(np.sign(x) * x**2)
    fs.append(x**3)
    fs.append(np.sign(x) * np.sqrt(abs(x)))
  return fs

# provided covariance matrix and average of market factor, we sample from a multivariate distribution
# we allow a seed to be passed for reproducibility
def simulate_market(f_ret_avg_pdf, f_ret_cov_pdf, seed):
  np.random.seed(seed = seed)
  return np.random.multivariate_normal(f_ret_avg_pdf, f_ret_cov_pdf)

# provided simulated market factors and a specific model for an instrument,
# predict the instrument return in 2 weeks time
def predict(fs, ps):
  s = ps[0]
  for i, f in enumerate(fs):
    s = s + ps[i + 1] * f
  return float(s)

# COMMAND ----------

# MAGIC %md
# MAGIC Here is a toy example of using the above code with an explicit random seed.

# COMMAND ----------

seed_init = 42
seeds = [seed_init + x for x in np.arange(0, 10)]
conditions = []
for seed in seeds:
  conditions.append(simulate_market(f_ret_avg, f_ret_cov, seed))

df = pd.DataFrame(conditions, columns=feature_names)
df['_seed'] = seeds
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP3` Run monte-carlo
# MAGIC
# MAGIC We'll use a Pandas UDF again to distribute training; note the code in our function is vanilla Python and Pandas.

# COMMAND ----------

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
# MAGIC We append the output to a Delta table so that _all_ of our trials are potentially available for querying.

# COMMAND ----------

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
# MAGIC ### Query a sample of the predictions

# COMMAND ----------

spark.read.table(trial_table).limit(100)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>