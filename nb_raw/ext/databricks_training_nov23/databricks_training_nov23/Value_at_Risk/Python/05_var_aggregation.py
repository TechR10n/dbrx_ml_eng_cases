# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Aggregation
# MAGIC
# MAGIC In this notebook, we demonstrate the versatile nature of our model carlo simulation on **Delta Lake**. Stored in its most granular form, analysts have the flexibility to slice and dice their data to aggregate value-at-risk on demand via a user aggregated defined function on **Spark SQL**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Configuration

# COMMAND ----------

# MAGIC %run "./Includes/setup"

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
from datetime import datetime, timedelta
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import functions as F

# COMMAND ----------

portfolio_table = 'portfolio'
stock_table = 'stock'
stock_return_table = 'stock_return'
market_table = 'market'
market_return_table = 'market_return'
trial_table = 'monte_carlo'

# COMMAND ----------

# MAGIC %scala
# MAGIC
# MAGIC //number of simulations
# MAGIC val runs = 50000
# MAGIC
# MAGIC //value at risk confidence
# MAGIC val confidenceVar = 95

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add a widget to show all run_dates

# COMMAND ----------

try:
  dbutils.widgets.remove('run')
except:
  print('No widget named [run]')

all_runs = sql("SELECT DISTINCT run_date FROM {}".format(trial_table)).toPandas()['run_date']
dbutils.widgets.dropdown("run", all_runs[0], all_runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Compute value at risk
# MAGIC
# MAGIC Here we'll calculate and visualize the VAR 95.

# COMMAND ----------

run = dbutils.widgets.get('run')

# aggregate monte carlo simulations
mc_df = (spark
  .read
  .table(trial_table)
  .filter(F.col('run_date') == run)
  .withColumnRenamed('trial', 'return')
  .groupBy('seed')
  .agg(F.sum('return').alias('return'))
  .select('return')
  .toPandas())

returns = mc_df['return']

# compute 95 value at risk
value_at_risk = np.quantile(returns, 5 / 100)
mc_df['exceeds'] = mc_df['return'] > value_at_risk

# extract normal distribution
avg = returns.mean()
std = returns.std()
x1 = np.arange(returns.min(),returns.max(),0.01)
y1 = stats.norm.pdf(x1, loc=avg, scale=std)
x2 = np.arange(returns.min(),value_at_risk,0.001)
y2 = stats.norm.pdf(x2, loc=avg, scale=std)

# plot value at risk
ax = mc_df.hist(column='return', bins=50, density=True, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
ax = ax[0]
for x in ax:
  
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    x.axvline(x=value_at_risk, color='r', linestyle='dashed', linewidth=1)
    x.fill_between(x2, y2, zorder=3, alpha=0.4)
    x.plot(x1, y1, zorder=3)
    
    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    x.text(value_at_risk - 0.2, 1, "VAR(95) = {:2f}".format(value_at_risk), rotation=90)
    x.set_title('')
    x.set_ylabel('')
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Slice and dice value at risk
# MAGIC
# MAGIC We'll define and register a Scala UDF to calculate VAR. This will allow analysts to query our data in an ad-hoc fashion.

# COMMAND ----------

# MAGIC %scala
# MAGIC
# MAGIC import org.apache.spark.sql.SparkSession
# MAGIC import org.apache.spark.sql.expressions.MutableAggregationBuffer
# MAGIC import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
# MAGIC import org.apache.spark.sql.Row
# MAGIC import org.apache.spark.sql.types._
# MAGIC
# MAGIC class ValueAtRisk(n: Int) extends UserDefinedAggregateFunction {
# MAGIC
# MAGIC   // This is the input fields for your aggregate function.
# MAGIC   override def inputSchema: org.apache.spark.sql.types.StructType = StructType(StructField("value", DoubleType) :: Nil)
# MAGIC
# MAGIC   // This is the internal fields you keep for computing your aggregate.
# MAGIC   override def bufferSchema: StructType = StructType(Array(StructField("worst", ArrayType(DoubleType))))
# MAGIC
# MAGIC   // This is the output type of your aggregatation function.
# MAGIC   override def dataType: DataType = DoubleType
# MAGIC
# MAGIC   // The order we process dataframe does not matter, the worst will always be the worst
# MAGIC   override def deterministic: Boolean = true
# MAGIC
# MAGIC   // This is the initial value for your buffer schema.
# MAGIC   override def initialize(buffer: MutableAggregationBuffer): Unit = {
# MAGIC     buffer(0) = Seq.empty[Double]
# MAGIC   }
# MAGIC
# MAGIC   // This is how to update your buffer schema given an input.
# MAGIC   override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
# MAGIC     buffer(0) = buffer.getAs[Seq[Double]](0) :+ input.getAs[Double](0)
# MAGIC   }
# MAGIC
# MAGIC   // This is how to merge two objects with the bufferSchema type.
# MAGIC   // We only keep worst N events
# MAGIC   override def merge(buffer: MutableAggregationBuffer, row: Row): Unit = {
# MAGIC     buffer(0) = (buffer.getAs[Seq[Double]](0) ++ row.getAs[Seq[Double]](0)).sorted.take(n)
# MAGIC   }
# MAGIC
# MAGIC   // This is where you output the final value, given the final value of your bufferSchema.
# MAGIC   // Our value at risk is best of the worst n overall
# MAGIC   override def evaluate(buffer: Row): Any = {
# MAGIC     return buffer.getAs[Seq[Double]](0).sorted.last
# MAGIC   }
# MAGIC
# MAGIC }
# MAGIC
# MAGIC // Assume we've generated 50,000 monte-carlo simulations for each instrument
# MAGIC val numRecords = runs
# MAGIC
# MAGIC // We want to compute Var(95)
# MAGIC val confidence = confidenceVar
# MAGIC
# MAGIC // So the value at risk is the best of the worst N events 
# MAGIC val n = (100 - confidence) * numRecords / 100
# MAGIC
# MAGIC // Register UADFs
# MAGIC val valueAtRisk = new ValueAtRisk(n)
# MAGIC spark.udf.register("VALUE_AT_RISK", new ValueAtRisk(n))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Total number of simulations

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM monte_carlo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Show evolution of value at risk

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW total_var AS 
# MAGIC SELECT 
# MAGIC   t.run_date AS day, 
# MAGIC   VALUE_AT_RISK(t.return) AS value_at_risk
# MAGIC FROM 
# MAGIC   (
# MAGIC   SELECT 
# MAGIC     m.run_date, 
# MAGIC     sum(m.trial) AS return
# MAGIC   FROM
# MAGIC     monte_carlo m
# MAGIC   GROUP BY
# MAGIC     m.run_date, 
# MAGIC     m.seed 
# MAGIC   ) t
# MAGIC GROUP BY 
# MAGIC   t.run_date
# MAGIC ORDER BY t.run_date ASC;
# MAGIC
# MAGIC SELECT * FROM total_var

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta Lake can use optimize and Z-order to speed up tables with many small files

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE monte_carlo ZORDER BY (country, industry)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look at risk exposure by country

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TEMPORARY VIEW country_var AS 
# MAGIC SELECT 
# MAGIC   t.run_date AS day, 
# MAGIC   LOWER(t.country) AS country,
# MAGIC   VALUE_AT_RISK(t.return) AS value_at_risk
# MAGIC FROM 
# MAGIC   (
# MAGIC   SELECT 
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.country, 
# MAGIC     sum(m.trial) AS return
# MAGIC   FROM
# MAGIC     monte_carlo m
# MAGIC   GROUP BY
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.country
# MAGIC   ) t
# MAGIC GROUP BY 
# MAGIC   t.run_date, 
# MAGIC   t.country
# MAGIC ORDER BY t.run_date ASC;
# MAGIC
# MAGIC SELECT * FROM country_var;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Focus in on Peru, aggregating by industry

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW peru_var AS 
# MAGIC SELECT 
# MAGIC   t.run_date AS day, 
# MAGIC   LOWER(t.industry) AS industry,
# MAGIC   VALUE_AT_RISK(t.return) AS value_at_risk
# MAGIC FROM 
# MAGIC   (
# MAGIC   SELECT 
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.industry, 
# MAGIC     sum(m.trial) AS return
# MAGIC   FROM
# MAGIC     monte_carlo m
# MAGIC   WHERE
# MAGIC     m.country = 'PERU'
# MAGIC   GROUP BY
# MAGIC     m.run_date, 
# MAGIC     m.seed, 
# MAGIC     m.industry
# MAGIC   ) t
# MAGIC GROUP BY 
# MAGIC   t.run_date, 
# MAGIC   t.industry
# MAGIC ORDER BY t.run_date ASC;
# MAGIC
# MAGIC CREATE OR REPLACE TEMPORARY VIEW peru_var_contribution AS
# MAGIC SELECT c.day, c.industry, c.value_at_risk / t.value_at_risk AS contribution
# MAGIC FROM peru_var c
# MAGIC JOIN (
# MAGIC   SELECT day, SUM(value_at_risk) AS value_at_risk
# MAGIC   FROM peru_var
# MAGIC   GROUP BY day
# MAGIC ) t
# MAGIC WHERE c.day = t.day;
# MAGIC
# MAGIC SELECT 
# MAGIC   c.day,
# MAGIC   c.industry,
# MAGIC   ABS(c.contribution * t.value_at_risk) AS contribution
# MAGIC FROM peru_var_contribution c
# MAGIC JOIN total_var t
# MAGIC WHERE t.day = c.day
# MAGIC ORDER BY t.day;
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>