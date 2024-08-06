-- Databricks notebook source
-- INCLUDE_HEADER_TRUE
-- INCLUDE_FOOTER_TRUE

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #  Managing external storage
-- MAGIC
-- MAGIC In this notebook you will learn how to:
-- MAGIC * Create a storage credential and external location
-- MAGIC * Control access to files using an external location
-- MAGIC * Create an external table
-- MAGIC * Compare attributes and behaviors of managed versus external tables

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Prerequisites
-- MAGIC
-- MAGIC If you would like to follow along with this lab, you must:
-- MAGIC * Have account admin capability in order to create a storage credential and external location
-- MAGIC * Have cloud resources (storage location and credentials for accessing the storage) to support an external location, which can be provided by your cloud administrator
-- MAGIC * Complete the procedures outlined in the following demos from the *Overview of Unity Catalog* course:
-- MAGIC     * *Managing Account Identities* (specifically, you need an *analysts* group containing another user with Databricks SQL access)
-- MAGIC     * *Managing SQL Warehouses* (specfically, you need a SQL warehouse to which the user mentioned above has access)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Setup
-- MAGIC
-- MAGIC Run the following cell to perform some setup. In order to avoid conflicts in a shared training environment, this will create a uniquely named catalog with schema exclusively for your use.

-- COMMAND ----------

-- MAGIC %run ./Includes/3-Managing-external-storage-setup

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating and managing a storage credential
-- MAGIC
-- MAGIC Storage credentials enable Unity Catalog to connect to external cloud storage locations and access files there, either for direct file access or for use with external tables. 
-- MAGIC
-- MAGIC We need a few cloud-specific resources in order to create one. Your cloud administrator can prepare these following these procedures:
-- MAGIC
-- MAGIC * For AWS, we must define an IAM role authorizing access to the S3 bucket, and we will need the ARN for that role. Refer to <a href="https://docs.databricks.com/data-governance/unity-catalog/create-tables.html#create-an-external-table" target="_blank">this document</a> for more details.
-- MAGIC * For Azure, we require a directory and application ID, and a client secret of a service principal that has been granted the **Azure Blob Contributor** role on the storage location. Refer to <a href="https://docs.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/create-tables#create-an-external-table" target="_blank">this document</a> for more details.
-- MAGIC
-- MAGIC Assuming we have the cloud resources we need, let's create the storage credential.
-- MAGIC
-- MAGIC 1. Open the **Data** page.
-- MAGIC 1. Click *Storage Credentials*.
-- MAGIC 1. Let's click **Create credential**.
-- MAGIC 1. Now let's specify the name. This must be unique for storage credentials in the metastore. The setup we ran earlier generated a unique name, so we can copy that generated value from the table output.
-- MAGIC 1. Paste the cloud credential as obtained from your cloud administrator (IAM role ARN or access connector resource ID, depending on which cloud you're on).
-- MAGIC 1. Let's click **Create**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's look at the properties of the storage credential using SQL.

-- COMMAND ----------

DESCRIBE STORAGE CREDENTIAL ${da.storage_credential}

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating and managing external locations
-- MAGIC
-- MAGIC Storage credentials represent a connection to an external cloud storage container and support a set of privileges that allows us to do full access control. However, any privileges granted on a storage credential apply to the entire container.
-- MAGIC
-- MAGIC It's fairly typical to subdivide such containers into subtrees for different purposes. If we do this, then it's also desirable to be able to control access to each of these subdivisions - something storage credential cannot do since they only apply to the entire thing.
-- MAGIC
-- MAGIC For this reason, we have external locations. External locations build on storage credentials and additionally specify a path within the referenced storage container, providing us with a construct that allows us to do access control at the level of files or folders.
-- MAGIC
-- MAGIC In order to define an external location, we need some additional information.
-- MAGIC
-- MAGIC * For AWS, we require the full S3 URL. This includes the bucket name and path within the container. Refer to <a href="https://docs.databricks.com/data-governance/unity-catalog/manage-external-locations-and-credentials.html#create-an-external-location" target="_blank">this document</a> for more details.
-- MAGIC * For Azure, we require the storage container path. Refer to <a href="https://docs.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/create-tables#create-an-external-location" target="_blank">this document</a> for more details.
-- MAGIC
-- MAGIC Let's create the external location.
-- MAGIC
-- MAGIC 1. On the **Data** page, click **External Locations**.
-- MAGIC 1. Let's click **Create location**.
-- MAGIC 1. Now let's specify the name. Just as with storage credentials, this must be unique for external locations in the metastore. The setup we ran earlier generated a unique name, so we can copy that generated value from the table output.
-- MAGIC     Note: an external location can have the same name as a storage credential (like we are doing here). Just keep in mind that in a production environment, you're probably not going to want to do this since there may often be more than one external location referencing a storage credential.
-- MAGIC 1. Paste the URL or path that fully qualifies the location you want to represent.
-- MAGIC 1. Let's select the storage credential representing the storage to which this external location pertains.
-- MAGIC 1. Let's click **Create**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's look at the properties of the external location using SQL.

-- COMMAND ----------

DESCRIBE EXTERNAL LOCATION ${da.external_location}

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Pay attention to the *url* value from the output of the previous cell. This will become important shortly.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Granting file access
-- MAGIC
-- MAGIC As a recap, storage credentials and external locations support a privilege set that allows us to govern access to the files stored in those locations. These privileges and their associated capabilities include:
-- MAGIC * **READ FILES**: directly read files from this location
-- MAGIC * **WRITE FILES**: directly write files to this location
-- MAGIC * **CREATE TABLE**: establish a table based on files stored in this location

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's put our external location to use by writing out some files that we will govern. We'll write the files with some Python code that:
-- MAGIC 1. Retrieves the *url* value for the external location (we talked about this value a few moments ago)
-- MAGIC 2. Extends the URL with a subdirectory named *test.delta*
-- MAGIC 3. Uses that location to write a simple dataset in Delta format.
-- MAGIC
-- MAGIC Since we are the data owner for the external location that covers this destination, we can write files without any additional privilege grants.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC external_delta_table = spark.sql(f"DESCRIBE EXTERNAL LOCATION {da.external_location}").first()['url'] + '/test.delta'
-- MAGIC spark.conf.set("da.external_delta_table", external_delta_table)
-- MAGIC
-- MAGIC df = spark.range(20)
-- MAGIC (df.write
-- MAGIC    .format("delta")
-- MAGIC    .mode('overwrite')
-- MAGIC    .save(external_delta_table))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's use the **`LIST`** statement to look at the files we just wrote out.

-- COMMAND ----------

LIST '${da.external_delta_table}'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Accessing files as an analyst
-- MAGIC
-- MAGIC Now let's attempt to read these data files as a different user. Recall in the **Prerequisites** section of this lab, we made a reference to having a group called *analysts* containing another user.
-- MAGIC
-- MAGIC In this section, we'll run queries as that user to verify our configuration, and observe the impact when we make changes.
-- MAGIC
-- MAGIC To prepare for this section, **you will need to log in to Databricks using a separate browser session**. This could be a private session, a different profile if your browser supports profiles, or a different browser altogether. Do not merely open a new tab or window using the same browser session; this will lead to login conflicts.
-- MAGIC
-- MAGIC 1. In a separate browser session, <a href="https://accounts.cloud.databricks.com/workspace-select" target="_blank">log in to Databricks</a> using the analyst user credentials.
-- MAGIC 1. Switch to the **SQL** persona.
-- MAGIC 1. Go to the **Queries** page and click **Create query**.
-- MAGIC 1. Select the shared SQL warehouse that was created while following the *Creating compute resources for Unity Catalog access* demo.
-- MAGIC 1. Return to this notebook and continue following along. When prompted, we will be switching to the Databricks SQL session and executing queries.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC First, let's run the following cell to generate a fully qualified query statement that does the same **`LIST`** operation we did a few moments ago here.

-- COMMAND ----------

SELECT "LIST '${da.external_delta_table}'" AS Query

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Copy the query from the output above and paste it into a new query in the Databricks SQL session and run it. It won't work yet, because there is no privilege on the external location. Let's grant the appropriate privelege now.

-- COMMAND ----------

GRANT READ FILES ON EXTERNAL LOCATION ${da.external_location} TO analysts

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Repeat the query in the Databricks SQL environment, and with this grant in place the operation will succeed.
-- MAGIC
-- MAGIC If you followed along with the *Creating and governing data objects with Unity Catalog* lab and are wondering why we haven't mentioned the need for any **USAGE** grants, it's because here we're dealing with external locations. In terms of the object hierarchy, storage credentials and external locations sit on their own, unlike tables, views and functions that are contained by a schema. There is no container to grant **USAGE** on.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating external tables
-- MAGIC
-- MAGIC Another important application of storage credentials and external locations is to support external tables; that is, tables whose data resides in external storage.
-- MAGIC
-- MAGIC Let's create an external table based on the delta files created previously. The syntax for creating an external table is similar to a managed table, with the addition of a **`LOCATION`** that specifies the full cloud storage path containing the data files backing the table.
-- MAGIC
-- MAGIC Keep in mind that when creating external tables, two conditions must be met:
-- MAGIC * We must have appropriate permissions (**CREATE TABLE**) on the external location, but since we are the data owner, no permission is necessary.
-- MAGIC * We must have appropriate permissions on the schema and catalog (**USAGE** and **CREATE**) in which we are creating the table. Again, we are the data owner, so no permissions are necessary.

-- COMMAND ----------

CREATE TABLE test_external
  LOCATION '${da.external_delta_table}'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's click the **Data** icon in the left sidebar to witness the effect of the above statement. We see that there is a new table contained within the *external_storage* schema. Let's quickly examine the contents of the table. As data onwers, this query runs without issue.

-- COMMAND ----------

SELECT * FROM test_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Grant access to an external table
-- MAGIC
-- MAGIC Once an external table is created, access control works the same way as it does for a managed table. Let's see this in action now.
-- MAGIC
-- MAGIC Let's run the following cell to output a query statement that reads the external table. Copy and paste the output into a new query within the Databricks SQL environment, and run the query.

-- COMMAND ----------

SELECT "SELECT * FROM ${da.catalog}.${da.schema}.test_external" AS Query

-- COMMAND ----------

-- MAGIC %md
-- MAGIC This operation fails. Yet a moment ago, there were no issues reading the underlying files. Why? Because now we're treating this as a table, and now we're subject to the security model for tables, which requires:
-- MAGIC * **SELECT** on the table
-- MAGIC * **USAGE** on the schema
-- MAGIC * **USAGE** on the catalog
-- MAGIC
-- MAGIC Let's address this now with the following three **`GRANT`** statements.

-- COMMAND ----------

GRANT SELECT ON TABLE test_external to analysts;
GRANT USAGE ON SCHEMA ${da.schema} to analysts;
GRANT USAGE ON CATALOG ${da.catalog} to analysts

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Re-run the query in the Databricks SQL session, and now this will succeed.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Comparing managed and external tables
-- MAGIC
-- MAGIC Let's compare managed and external tables.
-- MAGIC
-- MAGIC First, let's create a managed copy of the external table using **`CREATE TABLE AS SELECT`**.

-- COMMAND ----------

CREATE TABLE test_managed AS
  SELECT * FROM test_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's examine the contents of *test_managed*. The resuls are identical to those earlier when we queried *test_external*, since the table data was copied.

-- COMMAND ----------

SELECT * FROM test_managed

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's do a comparison of the two tables using **`DESCRIBE EXTENDED`**.

-- COMMAND ----------

DESCRIBE EXTENDED test_managed

-- COMMAND ----------

DESCRIBE EXTENDED test_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC For the most part, the tables are similar, though there are some difference, particularly in *Type* and *Location*.
-- MAGIC
-- MAGIC We all know that if we drop a managed table like *test_managed* and recreate it (without using a **`CREATE TABLE AS SELECT`** statement, of course) the new table will be empty. But, let's try it out to see. First, let's run the following to obtain the command needed to recreate *test_managed*.

-- COMMAND ----------

SHOW CREATE TABLE test_managed

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's drop the managed table.

-- COMMAND ----------

DROP TABLE test_managed

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's recreate the table. You can either run the code as it is below or replace it entirely with the equivalent code from *createtab_stmt* output above.

-- COMMAND ----------

CREATE TABLE ${da.catalog}.${da.schema}.test_managed (id BIGINT)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's look at the contents.

-- COMMAND ----------

SELECT * FROM test_managed

-- COMMAND ----------

-- MAGIC %md
-- MAGIC It shouldn't come as a surprise that the new table is empty.
-- MAGIC
-- MAGIC Now let's now try this for our external table, *test_external*. Let's drop it.

-- COMMAND ----------

DROP TABLE test_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's repeat the command from earlier to recreate *test_external*.

-- COMMAND ----------

CREATE TABLE test_external
  LOCATION '${da.external_delta_table}'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's look at the contents of the table.

-- COMMAND ----------

SELECT * FROM test_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We see here, the data persisted because the underlying data files remained intact after the table was dropped. This demonstrates a key difference in behavior between managed and external tables. With a managed table, data is discarded along with the metadata when the table is dropped. But with an external table, the data is left alone since it is unmanaged. Therefore it is retained when you recreate the table using the same location.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Clean up
-- MAGIC Run the following cell to remove the catalog that was created, along with the storage credential and external location.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC da.cleanup()
