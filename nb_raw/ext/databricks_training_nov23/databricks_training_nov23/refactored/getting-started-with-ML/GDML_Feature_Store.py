# Databricks notebook source
# MAGIC %md
# MAGIC # Getting Started with Databricks ML
# MAGIC <h2>Module 2: Databricks ML Data Preparation with Feature Store</h2>
# MAGIC <h3>Use Feature Store to create a feature table.</h3>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Case Explanation
# MAGIC The example for this content involves using a dataset of retail customers. The feature table created in this demonstration can be used later to build a model to predict the number of units a customer will purchase.
# MAGIC
# MAGIC ### Classroom Setup
# MAGIC ~~Before proceeding with this demonstration, you need to run the `Classroom-Setup` notebook to initialize the environment.~~

# COMMAND ----------

# MAGIC %md
# MAGIC ~~run "./Includes/Classroom-Setup"~~

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Data
# MAGIC
# MAGIC This step imports the data needed for this demonstration. This will be a dataset containing existing customer data.

# COMMAND ----------

input_path = "dbfs:/databricks_courseware/get-started-with-databricks-for-machine-learning/v01/retail/customers.csv"

# COMMAND ----------

customers_df = spark.read.csv(input_path, 
                              header=True)
display(customers_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Featurization
# MAGIC
# MAGIC The `customers_df` DataFrame is relatively clean, but there are some categorical features that need to be converted to numeric features for modeling.
# MAGIC
# MAGIC These features include:
# MAGIC
# MAGIC * **`state`**
# MAGIC * **`loyalty_segment`**
# MAGIC
# MAGIC #### Create `compute_features` Function
# MAGIC
# MAGIC Pandas DataFrames are familiar to most data scientists, so this demonstration uses the Pandas API on Spark library to one-hot encode these categorical features.
# MAGIC
# MAGIC Note that we are creating a function to perform these computations. We'll use the function to refer to this set of instructions when creating the feature table.

# COMMAND ----------

import pyspark.pandas as ps

def compute_features(spark_df):
    """
    Takes in a Spark Dataframe and returns a Pandas API on Spark Dataframe 
    that has one hot encoded columns, along with string clean up.
    Arguments:
    spark_df = a Spark Dataframe
    """
    
    df = ps.DataFrame(spark_df)
    
    # Sweeping drop to remove multiple instances of customers
    df = df.drop_duplicates(subset='customer_id')
    
    # One-Hot Encode the categorical columns
    ohe_ps_df = ps.get_dummies(
        df, 
        columns=["state", "loyalty_segment"],
        dtype="float64"
    )
    
    # Filter columns to just features for model to be trained on
    states = [colmn for colmn in ohe_ps_df.columns if "state_" in colmn]
    loyalty_segments = [colmn for colmn in ohe_ps_df.columns if "loyalty_segment" in colmn]
    states.extend(loyalty_segments)
    states.extend(["units_purchased", "customer_id"])
    
    ohe_ps_df = ohe_ps_df[states]
    
    # Clean datatypes
    ohe_ps_df['units_purchased'] = ohe_ps_df['units_purchased'].astype(int)
    
    return ohe_ps_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compute Features
# MAGIC
# MAGIC Next, use the featurize function `compute_features` to create a DataFrame of the features.

# COMMAND ----------

features_df = compute_features(customers_df)
display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Table
# MAGIC
# MAGIC Next, use the DataFrame **`features_df`** to create a feature table using Feature Store.
# MAGIC
# MAGIC #### Instantiate the `FeatureStoreClient`
# MAGIC
# MAGIC The first step is to instantiate the Feature Store client using `FeatureStoreClient()`.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Create the Feature Table
# MAGIC
# MAGIC Next, use the `feature_table` operation to register the DataFrame as a Feature Store table.
# MAGIC
# MAGIC To do this, provide the following:
# MAGIC
# MAGIC 1. The `name` of the database and table where we want to store the feature table
# MAGIC 1. The `keys` for the table
# MAGIC 1. The `schema` of the table
# MAGIC 1. A `description` of the contents of the feature table
# MAGIC
# MAGIC This creates the feature table, but the values in the DataFrame still need to be written to the table.
# MAGIC
# MAGIC Note: we're going to first cast the **`customer_id`** column with type **`integer`** before we set it as our primary key for our feature store. We're doing this because in a later noteboook, we're going to read our feature store table back in with the customer_id column having type integer!

# COMMAND ----------

features_df['customer_id'] = features_df['customer_id'].astype(int)

# COMMAND ----------

fs.create_table(
  name="customer_features",
  primary_keys=["customer_id"],
  schema=features_df.spark.schema(),
  description="This customer-level table contains one-hot encoded and numeric features to predict the number of units purchased by a customer."
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, write the records from **`features_df`** to the feature table.

# COMMAND ----------

fs.write_table(df=features_df.to_spark(), name=f"{DA.schema_name}.customer_features", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC At this point in the demonstration, navigate to the Feature Store (available in the menu on the left) to view the newly created table.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>