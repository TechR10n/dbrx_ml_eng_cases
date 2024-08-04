# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Data ETL and EDA
# MAGIC
# MAGIC In this notebook, we use `yfinance` to download stock data for 40 equities in an hypothetical Latin America portfolio. We show how to use `pandas UDF` paradigm to distribute this process efficiently and store all of our output data as a **Delta Lake** table so that our data is analytic ready at every point in time.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Install `yfinance`
# MAGIC Use `%pip` to install `yfinance` to all nodes in the cluster, but only for this SparkSession.
# MAGIC
# MAGIC <img alt="Caution" title="Caution" style="vertical-align: text-bottom; position: relative; height:1.3em; top:0.0em" src="https://files.training.databricks.com/static/images/icon-warning.svg"/> This will require a restart of your Python kernel; as such, any variables or imports in your current notebook would be lost. We do this install first to avoid this.

# COMMAND ----------

# MAGIC %pip install yfinance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Classroom Setup
# MAGIC
# MAGIC Note that you can [pass keyword arguments to widgets](https://docs.databricks.com/notebooks/widgets.html#use-widgets-with-run) when using `%run`.

# COMMAND ----------

# MAGIC %run "./Includes/setup" $mode="reset"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Environment
# MAGIC
# MAGIC We'll import a number of libraries and define the table names we'll be using throughout our notebooks.

# COMMAND ----------

import yfinance as yf
import pandas as pd
import numpy as np
from io import StringIO
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf
from datetime import datetime, timedelta

portfolio_table = 'portfolio'
stock_table = 'stock'
stock_return_table = 'stock_return'
market_table = 'market'
market_return_table = 'market_return'

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Create our portfolio
# MAGIC
# MAGIC We'll be using a handful of Latin American companies for our fake portfolio.

# COMMAND ----------

portfolio = """
country,company,ticker,industry
CHILE,Banco de Chile,BCH,Banks
CHILE,Banco Santander-Chile,BSAC,Banks
CHILE,Compañía Cervecerías Unidas S.A.,CCU,Beverages
CHILE,Itaú CorpBanca,ITCB,Banks
CHILE,"Enersis, S.A.",ENIA,Electricity
CHILE,Enersis Chile SA Sponsored ADR,ENIC,Electricity
CHILE,"SQM-Sociedad Química y Minera de Chile, S.A.",SQM,Chemicals
COLOMBIA,Avianca Holdings S.A.,AVH,Travel & Leisure
COLOMBIA,BanColombia S.A.,CIB,Banks
COLOMBIA,Ecopetrol S.A.,EC,Oil & Gas Producers
COLOMBIA,Grupo Aval Acciones y Valores S.A,AVAL,Financial Services
MEXICO,"América Móvil, S.A.B. de C.V.",AMX,Mobile Telecommunications
MEXICO,América Móvil SAB de CV Sponsored ADR Class A,AMOV,Mobile Telecommunications
MEXICO,CEMEX S.A.B. de C.V. (CEMEX),CX,Construction & Materials
MEXICO,"Coca-Cola FEMSA, S.A.B. de C.V.",KOF,Beverages
MEXICO,"Controladora Vuela Compañía de Aviación, S.A.B. de C.V",VLRS,Travel & Leisure
MEXICO,"Fomento Económico Mexicano, S.A.B. de C.V. (FEMSA)",FMX,Beverages
MEXICO,"Grupo Aeroportuario del Pacífico, S.A.B. de C.V. (GAP)",PAC,Industrial Transportation
MEXICO,"Grupo Aeroportuario del Sureste, S.A. de C.V. (ASUR)",ASR,Industrial Transportation
MEXICO,"Grupo Financiero Santander México, S.A.B. de C.V",BSMX,Banks
MEXICO,"Grupo Simec, S.A. De CV. (ADS)",SIM,Industrial Metals & Mining
MEXICO,"Grupo Televisa, S.A.",TV,Media
MEXICO,"Industrias Bachoco, S.A.B. de C.V. (Bachoco)",IBA,Food Producers
PANAMA,"Banco Latinoamericano de Comercio Exterior, S.A.",BLX,Banks
PANAMA,"Copa Holdings, S.A.",CPA,Travel & Leisure
PERU,Cementos Pacasmayo S.A.A.,CPAC,Construction & Materials
PERU,Southern Copper Corporation,SCCO,Industrial Metals & Mining
PERU,Fortuna Silver Mines Inc.,FSM,Mining
PERU,Compañía de Minas Buenaventura S.A.,BVN,Mining
PERU,Graña y Montero S.A.A.,GRAM,Construction & Materials
PERU,Credicorp Ltd.,BAP,Banks
"""

portfolio_df = pd.read_csv(StringIO(portfolio))

# COMMAND ----------

# MAGIC %md
# MAGIC We'll save these as a Delta Lake table so that they're accessible from any notebook in our workspace.

# COMMAND ----------

(spark
  .createDataFrame(portfolio_df)
  .select('ticker', 'company', 'country', 'industry')
  .write
  .format('delta')
  .mode('overwrite')
  .saveAsTable(portfolio_table))

display(spark.read.table(portfolio_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP2` Download stock data
# MAGIC We'll use an open source Python package to download historical ticker data from Yahoo Finance, which we'll be using for our modeling purposes.
# MAGIC
# MAGIC We will apply this function as a [pandas Grouped Map](https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html#grouped-map) which enables the efficient application of pandas functionality to grouped data in a Spark DataFrame.

# COMMAND ----------

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
  
(spark
  .read
  .table(portfolio_table)
  .groupBy("ticker")
  .applyInPandas(fetch_tick, schema=schema)
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(stock_table))

display(spark.read.table(stock_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use widgets to access a specific stock

# COMMAND ----------

# dbutils.widgets.remove('stock')
tickers = spark.read.table(portfolio_table).select('ticker').toPandas()['ticker']
dbutils.widgets.dropdown('stock', 'AVAL', tickers)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display a sorted view of a stock

# COMMAND ----------

display(spark
    .read
    .table(stock_table)
    .filter(F.col('ticker') == dbutils.widgets.get('stock'))
    .orderBy(F.asc('date'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP3` Download market factors
# MAGIC
# MAGIC We'll be using overall market trends as our predictive variables.

# COMMAND ----------

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
    msft = yf.Ticker(tick)
    raw = msft.history(period="2y")
    # fill in missing business days
    idx = pd.date_range(raw.index.min(), raw.index.max(), freq='B')
    # use last observation carried forward for missing value
    pdf = raw.reindex(idx, method='pad')
    factors_df[factors[tick]] = pdf['Close'].copy()
        
# Pandas does not keep index (date) when converted into spark dataframe
factors_df['Date'] = idx

# Overwrite delta table (bronze) with information to date
(spark.createDataFrame(factors_df)
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(market_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP4` Compute daily log return
# MAGIC
# MAGIC We'll log transform the returns for our stocks, as well as those for our market indicators.

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

# overwrite log returns to market table (gold)
(spark.createDataFrame(df)
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(market_return_table))

# COMMAND ----------

# MAGIC %md
# MAGIC Note how `Window` is being used to create a tumbling window for our measurements.

# COMMAND ----------

# Create UDF for computing daily log returns
@udf("double")
def compute_return(first, close):
  return float(np.log(close / first))

# Apply a tumbling 1 day window on each instrument
window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)

# apply sliding window and take first element
# compute returns
# make sure we have corresponding dates in market factor tables
sdf = (spark.table(stock_table)
  .filter(F.col('close').isNotNull())
  .withColumn("first", F.first('close').over(window))
  .withColumn("return", compute_return('first', 'close'))
  .select('date', 'ticker', 'return')
  .join(spark.table(market_return_table), 'date')
  .select('date', 'ticker', 'return'))

# overwrite log returns to market table (gold)
(sdf.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(stock_return_table))

# COMMAND ----------

# MAGIC %md
# MAGIC Use Databricks built-in plotting to generate a histogram of the distributions below.

# COMMAND ----------

display(spark.table(stock_return_table).filter(F.col('ticker') == dbutils.widgets.get('stock')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP5` Ensure data consistency
# MAGIC So far, we only have 1 version of our tables, but Delta Lake allows us to see the historical operations that result in our versions, as well as query past versions of our data.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY market_return

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM market_return
# MAGIC VERSION AS OF 0
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>