# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Back Testing
# MAGIC
# MAGIC The Basel Committee specified a methodology for backtesting VaR. The 1 day VaR 99 results are to
# MAGIC be compared against daily P&L’s. Backtests are to be performed quarterly using the most recent 250
# MAGIC days of data. Based on the number of exceedances experienced during that period, the VaR
# MAGIC measure is categorized as falling into one of three colored zones:
# MAGIC
# MAGIC | Level   | Threshold                 | Results                       |
# MAGIC |---------|---------------------------|-------------------------------|
# MAGIC | Green   | Up to 4 exceedances       | No particular concerns raised |
# MAGIC | Yellow  | Up to 9 exceedances       | Monitoring required           |
# MAGIC | Red     | More than 10 exceedances  | VaR measure to be improved    |

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Configuration

# COMMAND ----------

# MAGIC %run "./Includes/setup"

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.SparkSession
# MAGIC import org.apache.spark.sql.expressions.MutableAggregationBuffer
# MAGIC import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
# MAGIC import org.joda.time.DateTime
# MAGIC import org.apache.spark.sql.Row
# MAGIC import org.apache.spark.sql.types._
# MAGIC import org.apache.spark.sql.functions._

# COMMAND ----------

# MAGIC %scala
# MAGIC val portfolio_table = "portfolio"
# MAGIC val stock_table = "stock"
# MAGIC val stock_return_table = "stock_return"
# MAGIC val market_table = "market"
# MAGIC val market_return_table = "market_return"
# MAGIC val trial_table = "monte_carlo"
# MAGIC
# MAGIC //number of simulations
# MAGIC val runs = 50000
# MAGIC
# MAGIC //value at risk confidence
# MAGIC val confidenceVar = 95

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register VAR UDF from last notebook

# COMMAND ----------

# MAGIC %scala
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
# MAGIC     buffer.getAs[Seq[Double]](0).sorted.last
# MAGIC   }
# MAGIC
# MAGIC }
# MAGIC
# MAGIC // Assume we've generated 100,000 monte-carlo simulations for each instrument
# MAGIC val numRecords = runs
# MAGIC
# MAGIC // We want to compute Var(99)
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
# MAGIC ## `STEP1` Compute value at risk
# MAGIC
# MAGIC Start by loading in our trials and calculating our 95 confidence var.

# COMMAND ----------

# MAGIC %scala
# MAGIC
# MAGIC val convertDate = udf((s: String) => {
# MAGIC   new java.text.SimpleDateFormat("yyyy-MM-dd").parse(s).getTime
# MAGIC })
# MAGIC
# MAGIC case class VarHistory(time: Long, valueAtRisk: Double)
# MAGIC
# MAGIC val historicalVars = sql(s"""
# MAGIC   SELECT t.run_date, VALUE_AT_RISK(t.return) AS valueAtRisk
# MAGIC   FROM 
# MAGIC     (
# MAGIC     SELECT m.run_date, m.seed, sum(m.trial) AS return
# MAGIC     FROM ${trial_table} m
# MAGIC     GROUP BY m.run_date, m.seed
# MAGIC     ) t
# MAGIC   GROUP BY 
# MAGIC     t.run_date
# MAGIC   """
# MAGIC   )
# MAGIC   .withColumn("time", convertDate(col("run_date")))
# MAGIC   .orderBy(asc("time"))
# MAGIC   .select("time", "valueAtRisk")
# MAGIC   .as[VarHistory]
# MAGIC   .collect()
# MAGIC   .sortBy(_.time)
# MAGIC   .reverse
# MAGIC
# MAGIC val historicalVarsB = spark.sparkContext.broadcast(historicalVars)
# MAGIC display(historicalVars.toList.toDF())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform an as-of join with historical returns

# COMMAND ----------

# MAGIC %scala
# MAGIC val asOfVar = udf((s: java.sql.Date) => {
# MAGIC   val historicalVars = historicalVarsB.value
# MAGIC   if(s.getTime < historicalVars.last.time) {
# MAGIC     Some(historicalVars.last.valueAtRisk)
# MAGIC   } else {
# MAGIC     historicalVarsB.value.dropWhile(_.time > s.getTime).headOption.map(_.valueAtRisk) 
# MAGIC   }
# MAGIC })
# MAGIC
# MAGIC val minDate = new java.sql.Date(new DateTime(historicalVars.last.time).minusDays(250).getMillis)
# MAGIC val maxDate = new java.sql.Date(historicalVars.head.time)
# MAGIC
# MAGIC val historicalReturns = spark
# MAGIC   .read
# MAGIC   .table(stock_return_table)
# MAGIC   .filter(col("date") >= lit(minDate))
# MAGIC   .filter(col("date") <= lit(maxDate))
# MAGIC   .groupBy("date")
# MAGIC   .agg(sum("return").as("return"))
# MAGIC   .withColumn("var", asOfVar(col("date")))
# MAGIC   .orderBy(asc("date"))
# MAGIC
# MAGIC display(historicalReturns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Extract breaches
# MAGIC
# MAGIC We'll calculate the number of breaches over a 250 day sliding window

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.expressions.Window
# MAGIC
# MAGIC val toTime = udf((s: java.sql.Date) => {
# MAGIC   s.getTime / 1000
# MAGIC })
# MAGIC
# MAGIC val windowSpec = Window.orderBy("time").rangeBetween(-3600 * 24 * 250, 0)
# MAGIC val countBreaches = udf((asOfVar: Double, returns: Seq[Double]) => {
# MAGIC   returns.count(_ < asOfVar)
# MAGIC })
# MAGIC
# MAGIC historicalReturns
# MAGIC   .withColumn("time", toTime(col("date")))
# MAGIC   .withColumn("returns", collect_list("return").over(windowSpec))
# MAGIC   .withColumn("count", countBreaches(col("var"), col("returns")))
# MAGIC   .drop("returns", "time")
# MAGIC   .createOrReplaceTempView("breaches")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot breaches

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt 

breaches = sql("SELECT date, return, CAST(var AS DOUBLE), `count` FROM breaches ORDER BY date ASC").toPandas()

# create pandas datetime dataframe
breaches.index = breaches['date']
breaches = breaches.drop(['date'], axis=1)

# detect breach severity
basel1 = breaches[breaches['count'] <= 4]
basel2 = breaches[(breaches['count'] > 4) & (breaches['count'] < 10)]
basel3 = breaches[breaches['count'] >= 10]

# plot it
f, (a0, a1) = plt.subplots(2, 1, figsize=(20,8), gridspec_kw={'height_ratios': [10,1]})

a0.plot(breaches.index, breaches['return'], color='#86bf91', label='returns')
a0.plot(breaches.index, breaches['var'], label="var99", c='red', linestyle='--')
a0.axhline(y=0, linestyle='--', alpha=0.2, color='#86bf91', zorder=1)
a0.title.set_text('VAR99 backtesting')
a0.set_ylabel('Daily log return')
a0.legend(loc="upper left")
a0.get_xaxis().set_ticks([])

a1.bar(basel1.index, 1, color='green', label='breaches', alpha=0.1, width=10, zorder=3)
a1.bar(basel2.index, 1, color='orange', label='breaches', alpha=0.1, width=10, zorder=5)
a1.bar(basel3.index, 1, color='red', label='breaches', alpha=0.1, width=10, zorder=10)
a1.get_yaxis().set_ticks([])
a1.set_xlabel('Date')

plt.subplots_adjust(wspace=0, hspace=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Investigate periods with 10 or more breaches

# COMMAND ----------

basel3.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wall St banks’ trading risk surges to highest since 2011
# MAGIC
# MAGIC [...] The top five Wall St banks’ aggregate “value-at-risk”, which measures their potential daily trading losses, soared to its highest level in 34 quarters during the first three months of the year, according to Financial Times analysis of the quarterly VaR high disclosed in banks’ regulatory filings
# MAGIC
# MAGIC [https://on.ft.com/2SSqu8Q](https://on.ft.com/2SSqu8Q)
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>