-- Databricks notebook source
-- INCLUDE_HEADER_TRUE
-- INCLUDE_FOOTER_TRUE

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Upgrading tables to Unity Catalog
-- MAGIC
-- MAGIC In this notebook you will learn how to:
-- MAGIC * Upgrade a table from the local Hive metastore to Unity Catalog using three different approaches
-- MAGIC * Understand the cases where each approach is appropriate

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Prerequisites
-- MAGIC If you would like to follow along with this notebook, you will need:
-- MAGIC * **USAGE** and **CREATE** permissions on the *main* catalog. If you have access to a differently named catalog, then update the value for *catalog* before proceeding to run the setup cells.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Setup
-- MAGIC
-- MAGIC Run the following cells to perform some setup. In order to avoid conflicts in a shared training environment, this will create a uniquely named schema exclusively for your use. This will also create an example source table called *movies* within the legacy Hive metatore. 

-- COMMAND ----------

-- MAGIC %python
-- MAGIC catalog = "main"

-- COMMAND ----------

-- MAGIC %run ./Includes/4-Upgrading-tables-setup

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exploring the source table
-- MAGIC
-- MAGIC As part of the setup, we now have a table called *movies*, residing in a user-specific schema of the Hive metastore. To make things easier, the schema name is stored in a Hive variable named **`da.schema`**. Let's preview the data stored in this table using that variable. Notice how the three-level namespaces makes referencing data objects in the Hive metastore seamless.

-- COMMAND ----------

SELECT * FROM hive_metastore.${da.schema}.movies LIMIT 10

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Setting up a destination
-- MAGIC
-- MAGIC With a source table in place, let's set up a destination in Unity Catalog.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating a schema
-- MAGIC
-- MAGIC In a production environment, it's likely you have a schema already set up at this point, but for this training exercise, let's create one now using the unique name generated as part of the setup.

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS ${da.catalog}.${da.schema}

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Upgrade methods
-- MAGIC
-- MAGIC There are a few different ways to upgrade a table. The method you choose will be driven primarily by which of the two following categories the upgrade falls into. In both cases we're upgrading metadata, but they differ in how data is treated:
-- MAGIC 1. Move data. In this case, we are moving the table data from where it currently resides (whether it be managed or external) to the managed storage of the destination metastore.
-- MAGIC 2. Leave data in place. In other words, upgrade an external table, leaving it as an external table.
-- MAGIC
-- MAGIC As managed tables in Unity Catalog carry a lot of benefits, Databricks recommends using them wherever possible; a scenario which is covered by the first option. That said, Databricks also recognizes that customers often need to use external tables, so the second option provides a low-cost option when upgrading massive tables without disturbing the data.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Moving table data to the metastore
-- MAGIC
-- MAGIC In this section, we cover two different options for upgrading tables, both of which will move table data to the metastore. Both options share the same pros and cons.
-- MAGIC
-- MAGIC The main advantages of moving data into the Unity Catalog metastore are as follows:
-- MAGIC * Managed tables in Unity Catalog bring additional benefits in terms of administration and performance
-- MAGIC * Allows for transformation of data enroute, in case you want to upgrade tables structures during the move to match evolved business requirements
-- MAGIC
-- MAGIC The main disadvantage to this approach is:
-- MAGIC * For massive tables, this can take time and possibly incur transfer cost

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Create Table As Select (CTAS) using CLONE
-- MAGIC
-- MAGIC This is the optimal method to use when migrating managed Delta tables. In Unity Catalog, all managed tables are Delta tables, however the Hive metastore supports other formats, so this method should only be used when the source table known to be Delta (which it is in this case). Let's perform the operation now creating a destination table named *movies_clone*.

-- COMMAND ----------

CREATE OR REPLACE TABLE ${da.catalog}.${da.schema}.movies_clone CLONE hive_metastore.${da.schema}.movies

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Create Table As Select (CTAS)
-- MAGIC
-- MAGIC When the source format is not Delta, a standard CTAS staement can be universally used as follows.

-- COMMAND ----------

CREATE OR REPLACE TABLE ${da.catalog}.${da.schema}.movies
  AS SELECT * FROM hive_metastore.`${da.schema}`.movies

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Applying tranformations during the upgrade
-- MAGIC
-- MAGIC Though upgrading a table to Unity Catalog is a simple operation, it's a great time to closely consider your tables structures and whether they still address your organization's business requirements that may have changed over time.
-- MAGIC
-- MAGIC The examples we saw earlier take an exact copy of the source table. Both approaches use the basic **`CREATE TABLE AS SELECT`** construct, but we can easily perform any transformations during the upgrade. For example, let's expand on the previous example to do the following tranformations:
-- MAGIC * Assign the name *idx* to the first column
-- MAGIC * Additionally select only the columns *title*, *year*, *budget* and *rating*
-- MAGIC * Convert *year* and *budget* to **INT** (replacing any instances of the string *NA* with 0)
-- MAGIC * Convert *rating* to **DOUBLE**

-- COMMAND ----------

CREATE OR REPLACE TABLE ${da.catalog}.${da.schema}.movies_transformed
AS SELECT
  _c0 AS idx,
  title,
  CAST(year AS INT) AS year,
  CASE WHEN
    budget = 'NA' THEN 0
    ELSE CAST(budget AS INT)
  END AS budget,
  CAST(rating AS DOUBLE) AS rating
FROM hive_metastore.${da.schema}.movies

-- COMMAND ----------

SELECT * FROM ${da.catalog}.${da.schema}.movies_transformed

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC ## Upgrading metadata only
-- MAGIC
-- MAGIC We have seen approaches that involve moving table data from wherever it is currently (be it external or managed) to the Unity Catalog metastore storage. However, in upgrading existing external tables, some may prefer to leave data in place. In this case, we can use the upgrade wizard that can be found in the Data Explorer user interface in <a href="/sql" target="_blank">Databricks SQL</a> .
-- MAGIC
-- MAGIC 1. Open the **Data** page in Databricks SQL.
-- MAGIC 1. Select the *hive_metastore* catalog.
-- MAGIC 1. Select the schema in accordance with the schema named in the setup output.
-- MAGIC 1. Select the external table *test_external*. The wizard is not capable of upgrading managed tables and will not provide the option to do this.
-- MAGIC 1. Click **Upgrade**.
-- MAGIC 1. Specify a destination catalog according to the catalog named in the setup output.
-- MAGIC 1. Select the schema in accordance with the schema named in the setup output. This is an identically named schema to the one found in *hive_metastore*.
-- MAGIC 1. Specify the owner, or leave it as the default, and click **Next**.
-- MAGIC 1. The final page gives us the option of running the output or reviewing the SQL query that will implement the upgrade operation.
-- MAGIC
-- MAGIC Because we do not have a storage credential set up for the storage location where the table data resides, this operation will not succeed. However this section illustrates the workflow.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Clean up
-- MAGIC Run the following cell to remove the resources that were created in this example.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC da.cleanup()
