# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Modeling
# MAGIC
# MAGIC In this notebook, we retrieve last 2 years worth of market indicator data to train a model that could predict our instrument returns. As our portfolio is made of 40 equities, we want to train 40 predictive models in parallel, collecting all weights into a single coefficient matrix for monte carlo simulations. We show how to have a more discipline approach to model development by leveraging **MLFlow** capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP0` Configuration

# COMMAND ----------

# MAGIC %run "./Includes/setup"

# COMMAND ----------

import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import mlflow

from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from datetime import datetime, timedelta

# COMMAND ----------

portfolio_table = 'portfolio'
stock_table = 'stock'
stock_return_table = 'stock_return'
market_table = 'market'
market_return_table = 'market_return'

# when do we train model
today_str = "2020-02-07"
today = F.to_date(F.lit(today_str)).cast(TimestampType())
mlflow.log_param('to_date', today_str)

# our predictive market factors
feature_names = ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES']

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP1` Access data
# MAGIC
# MAGIC We'll capture the starting versions of our tables as variables to log for reproducibility.

# COMMAND ----------

versions_m_df = sql("DESCRIBE HISTORY " + market_return_table).select("version")
delta_m_version = versions_m_df.toPandas()['version'].max()

versions_s_df = sql("DESCRIBE HISTORY " + stock_return_table).select("version")
delta_s_version = versions_s_df.toPandas()['version'].max()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load in stock and market factor returns

# COMMAND ----------

# retrieve historical tick data up to specified date
f_ret = spark.table(market_return_table).filter(F.col('date') <= today)
s_ret = spark.table(stock_return_table).filter(F.col('date') <= today)

# market factors easily fit in memory and are required to build normal distribution of returns
f_ret_pdf = f_ret.toPandas()
f_ret_pdf.index = f_ret_pdf['date']
f_ret_pdf = f_ret_pdf.drop(['date'], axis=1)
mlflow.log_metric('x_size', f_ret_pdf.size)
f_ret_pdf.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##`STEP2` Evaluate market factors
# MAGIC
# MAGIC Normally, we'd use a heatmap to confirm that our predictive variables are not correlated; here, we're not surprised to see high correlation between many of our market factors.

# COMMAND ----------

# we simply plot correlation matrix via pandas (market factors fit in memory)
# we assume market factors are not correlated (NASDAQ and SP500 are, so are OIL and TREASURY BONDS)
f_cor_pdf = f_ret_pdf.corr(method='spearman', min_periods=12)
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(f_cor_pdf, annot=True)
plt.savefig(f"{userhome.replace('dbfs:', '/dbfs')}/tmp/factor_correlation.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##`STEP3` Train a model for each instrument
# MAGIC
# MAGIC Note that our resultant DataFrame contains our training features as an array and our `y` variable as a separate column. We'll later filter this data by `ticker` to build a separate model for each of our stonks.

# COMMAND ----------

# create our feature set based on market factors and actual portfolio return
# in real life, we should obviously split set into training / testing
x_train = (f_ret
  .withColumn("features", F.array(feature_names))
  .dropna()
  .select('date', 'features')
  .join(s_ret, 'date'))

display(x_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Pandas UDF to train in parallel
# MAGIC
# MAGIC Because we're just training OLS, only the coefficients need to be logged to recreate our model. Note that with this pattern, we could substitute any algorithm here and make new predictions.
# MAGIC
# MAGIC Note that both Spark SQL and DataFrames can define nested schema using a simple `array<type>` string syntax.

# COMMAND ----------

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
models_path = f"{userhome.replace('dbfs:', '/dbfs')}/tmp/models.json"
models_df.to_json(models_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict the daily returns
# MAGIC We'll join our predictions back with our true values so we can get a snapshot of how our model is fitting our data.

# COMMAND ----------

# simply applying weight to each market factor feature
@udf("float")
def predict_udf(xs, ps):
  fs = featurize(xs) 
  s = ps[0]
  for i, f in enumerate(fs):
    s = s + ps[i + 1] * f
  return float(s)

# we read models created at previous step
models_df = spark.createDataFrame(pd.read_json(models_path))

# we join model for each return to compute prediction of return vs. actual
prediction_df = (x_train.join(models_df, ['ticker'])
  .withColumn("predicted", predict_udf(F.col('features'), F.col('weights')))
  .withColumnRenamed('return', 'actual')
  .select('ticker', 'date', 'predicted', 'actual'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute the sum of squared errors
# MAGIC Remember our values are still log transformed, and that this metric is being calculated against our training data.

# COMMAND ----------

@udf("float")
def sse_udf(p, a):
  return float((p - a)**2)

# compare expected vs. actual return
# sum square error per instrument
sse_df = (prediction_df
  .withColumn('sse', sse_udf(F.col('predicted'), F.col('actual')))
  .groupBy('ticker')
  .agg(F.sum('sse'))
  .toPandas())
 
# plot mean square error as accuracy of our model for each instrument
ax=sse_df.plot.bar(x='ticker', y='sum(sse)', rot=0, label=None, figsize=(24,5))
ax.get_legend().remove()
plt.title("Model SSE for each instrument")
plt.xticks(rotation=45)
plt.ylabel("sse")
plt.savefig(f"{userhome.replace('dbfs:', '/dbfs')}/tmp/model_sse.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Show predictive value for [Ecopetrol S.A.], Oil & Gas Producers in Columbia

# COMMAND ----------

df = prediction_df.filter(F.col('ticker') == "EC").toPandas()
plt.figure(figsize=(20,8))
plt.plot(df.date, df.actual)
plt.plot(df.date, df.predicted, color='green', linestyle='--')
plt.title('Log return of EC')
plt.ylabel('log return')
plt.xlabel('date')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## `STEP4` register model
# MAGIC We can use MLflow to register various parameters and artifacts. Note that by default, each Databricks notebook will integrate with MLflow as an experiment.

# COMMAND ----------

artifact_base = f"{userhome.replace('dbfs:', '/dbfs')}/tmp/"

mlflow.log_param('delta.version.market', delta_m_version)
mlflow.log_param('delta.version.stocks', delta_s_version)
mlflow.log_artifact(f'{artifact_base}model_sse.png')
mlflow.log_artifact(f'{artifact_base}factor_correlation.png')
mlflow.log_artifact(f'{artifact_base}models.json')
mlflow.end_run()


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>