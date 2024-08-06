-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Integrating External Storage
-- MAGIC
-- MAGIC In this demo you will learn how to:
-- MAGIC * Connect Databricks to additional S3 buckets
-- MAGIC * Read from and write to external storage

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Prerequisites
-- MAGIC
-- MAGIC If you would like to follow along with this demo, you will need:
-- MAGIC * Administrator access to your AWS console, with the ability to create buckets and IAM roles
-- MAGIC * Account administrator capabilities in your Databricks account

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Overview
-- MAGIC
-- MAGIC There are some use cases that require external storage. Primarily these use cases are related to the storage of data assets outside the metastore bucket and the workspace root bucket. These scenarios include:
-- MAGIC
-- MAGIC * Migrating large datasets from a traditional Hive metastore to Unity Catalog. If there is a need to avoid the cost and time required to move this data, accessing them in place as a collection of external tables provides a viable alternative.
-- MAGIC * External writers: that is, ongoing processes outside of your Databricks deployment that write to the storage.
-- MAGIC * Strict business or regulalatory requirements that impose a hierarchy or naming convention on your storage containers
-- MAGIC * Hard requirement for data isolation at the infrastructural level
-- MAGIC * Requirement to manage data in a format that isn't Delta
-- MAGIC
-- MAGIC In this lab, we'll walk how to set up Databricks and Unity Catalog to work with external storage in an S3 bucket that we'll create.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating a bucket
-- MAGIC
-- MAGIC In a real world scenario, you likely already have a bucket you want to connect to, but for the purposes of this lab, let's create an S3 bucket that we will connect.
-- MAGIC
-- MAGIC 1. In the AWS S3 console, click **Create bucket**.
-- MAGIC 1. Let's specify a name. When choosing your own names, be mindful to not include dots in your names. Bucket names must also be globally unique. In this example we use *dbacademy-test-external-bucket*, but you should include a suffix or prefix that uniquely ties the name to your organization; for example, replace *dbacademy* with your domain name (using hyphens instead of dots).
-- MAGIC 1. Let's choose a region; note that this does not have to coincide with the worksapce or metastore. To prove this point, let's choose a different region, like *us-west-1*.
-- MAGIC 1. For this example, let's accept the default settings for the rest, and create the bucket.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating an IAM role for storage access
-- MAGIC
-- MAGIC In order to access data stored in the bucket, there needs to be an IAM role that authorizes reading and writing. You reference that IAM role when you create the storage credential.
-- MAGIC
-- MAGIC 1. In the AWS IAM console, let's select **Roles**.
-- MAGIC 1. Click **Create role**.
-- MAGIC 1. Choose **AWS account**. This will help us set up a cross-account trust relationship that allows Unity Catalog (a Databricks servce) to access the bucket directly.
-- MAGIC    * Select **Another AWS account**.
-- MAGIC    * For **Account ID**, let's substitute in the Databricks account ID, *414351767826*.
-- MAGIC    * Select **Require external ID**.
-- MAGIC    * For **External ID**, let's paste our Databricks account ID. We can easily get this from the user menu in the account console.
-- MAGIC    * Now let's click **Next** until we get to the final page.
-- MAGIC    * Let's assign the name for our role (use *dbacademy-test-external-storage-role*).
-- MAGIC    * Click **Create role**.
-- MAGIC 1. We're not quite finished yet; we still need to do some additional configuration on the trust policy. So let's locate the role we just created and select it. Take note of the **ARN**; we will need that in a couple of places.
-- MAGIC    * In the **Trust relationships** tab, let's click **Edit trust policy**.
-- MAGIC    * In the *Principal* block, replace *root* with **`role/unity-catalog-prod-UCMasterRole-14S5ZJVKOTYTL`**. This is a static value that references a role created by Databricks for Unity Catalog's use.
-- MAGIC    * Click the **Add** button to add a principal.
-- MAGIC    * Select **IAM Roles** for the **Principal type**.
-- MAGIC    * In the **ARN** field, paste the ARN for the role we just created and are now editing. This additional principal is a self-reference that allows the role to be self-assuming.
-- MAGIC    * Click **Add principal**.
-- MAGIC    * Finally, let's click **Update policy**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Create and attach a policy
-- MAGIC
-- MAGIC We'll need to attach a policy to the IAM role outlining the resources and actions permitted; let's do that now.
-- MAGIC
-- MAGIC 1. Still in the IAM console, let's go to **Policies**.
-- MAGIC 1. Click **Create policy**.
-- MAGIC 1. Click the **JSON** tab and replace the default policy with the following:
-- MAGIC     ```
-- MAGIC     {
-- MAGIC       "Version": "2012-10-17",
-- MAGIC       "Statement": [
-- MAGIC         {
-- MAGIC           "Action": [
-- MAGIC             "s3:GetObject",
-- MAGIC             "s3:PutObject",
-- MAGIC             "s3:DeleteObject",
-- MAGIC             "s3:ListBucket",
-- MAGIC             "s3:GetBucketLocation",
-- MAGIC             "s3:GetLifecycleConfiguration",
-- MAGIC             "s3:PutLifecycleConfiguration"
-- MAGIC           ],
-- MAGIC           "Resource": [
-- MAGIC             "arn:aws:s3:::<BUCKET>/*",
-- MAGIC             "arn:aws:s3:::<BUCKET>"
-- MAGIC           ],
-- MAGIC           "Effect": "Allow"
-- MAGIC         },
-- MAGIC         {
-- MAGIC           "Action": [
-- MAGIC             "sts:AssumeRole"
-- MAGIC           ],
-- MAGIC           "Resource": [
-- MAGIC             "<ROLE_ARN>"
-- MAGIC           ],
-- MAGIC           "Effect": "Allow"
-- MAGIC         }
-- MAGIC      ]
-- MAGIC    }
-- MAGIC    ```
-- MAGIC 1. Now let's customize the policy.
-- MAGIC    * Replace instances of **`<BUCKET>`** with the name of the bucket we created (*dbacademy-test-external-bucket*)
-- MAGIC    * Replace **`<ROLE_ARN>`** with the ARN of *dbacademy-test-external-storage-role* (to which we are about to attach this policy).
-- MAGIC 1. Let's click through accepting the default settings for the rest and specifying a suitable name (use *dbacademy-test-external-storage-policy*), then create the policy.
-- MAGIC 1. Now let's locate and select *dbacademy-test-external-storage-role*.
-- MAGIC 1. In the **Permissions** tab, click **Add permissions > Attach policies**.
-- MAGIC 1. Let's locate and select *dbacademy-test-external-storage-policy*, then click **Attach policies**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating a storage credential
-- MAGIC
-- MAGIC A storage credential is a Databricks construct that lives in the metastore. It represents an authentication and authorization mechanism for accessing data stored in your cloud storage. Essentially, it's a wrapper for an IAM role used to access that storage. Storage credentials can be access-controlled like other data objects in the metastore.
-- MAGIC
-- MAGIC We create storage credentials in a workspace that is assigned to the metastore. Only account administrators can create them. Let's create this now.
-- MAGIC
-- MAGIC 1. Let's log in to a deployed workspace as a user with account administrator privileges.
-- MAGIC 1. Go to the **Data** page.
-- MAGIC 1. Open the **Storage Credentials** panel, and click **Create credential**.
-- MAGIC 1. Specify a name (let's use *dbacademy-test-external-storage-credential*).
-- MAGIC 1. Specify the ARN of *dbacademy-test-external-storage-role*.
-- MAGIC 1. Now, let's click **Create**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating external locations
-- MAGIC
-- MAGIC While storage credentials are technically sufficient to provide access control on cloud storage, it's very coarse because it applies to the entire container. Often we'd like to exercise access controls on individual tables, files or folders - all of which represent subdivisions of the container.
-- MAGIC
-- MAGIC This is where external locations become useful. An external location is an object that combines a cloud storage path with a storage credential. Since each external location can be individually access controlled, we can then control access to portions of the container represented by the storage credential.
-- MAGIC
-- MAGIC Like storage credentials, we create external locations in a workspace that is assigned to the metastore. You do not have to be an account administrator to create an external location, but you do have to be a metastore administrator, or you must have the *CREATE EXTERNAL LOCATION* permission on the associated storage credential.
-- MAGIC
-- MAGIC 1. Still in the **Data** page, open the **External Locations** panel, and click **Create location**.
-- MAGIC 1. Specify a name (let's use *dbacademy-test-external-location*).
-- MAGIC 1. Specify the complete path (including the bucket name and a path within the bucket, if desired).
-- MAGIC 1. Select the storage credential.
-- MAGIC 1. Now, let's click **Create**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Testing external storage
-- MAGIC
-- MAGIC Now let's test what we've created. First of all, let's examine the storage credential and external location we've created by executing the following commands (you'll need to create a cluster to execute these commands).
-- MAGIC
-- MAGIC First let's list the storage credentials.

-- COMMAND ----------

SHOW STORAGE CREDENTIALS

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Notice how this lists the storage credential we explicitly created, as well as an additional entry. This second entry corresponds to the internal storage credential created as part of the metastore to enable access to the underlying metastore bucket.
-- MAGIC
-- MAGIC We can obtain details on individual credentials using **`DESCRIBE STORAGE CREDENTIAL`** as follows.

-- COMMAND ----------

DESCRIBE STORAGE CREDENTIAL `dbacademy-test-external-storage-credential`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's list the external locations.

-- COMMAND ----------

SHOW EXTERNAL LOCATIONS

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Likewise, we can obtain details on individual external locations using **`DESCRIBE EXTERNAL LOCATION`** as follows.

-- COMMAND ----------

DESCRIBE EXTERNAL LOCATION `dbacademy-test-external-location`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating and managing external tables
-- MAGIC Now let's create an external table using the external storage we set up. To do this, we'll duplicate a sample dataset provided by Databricks in a simple **`CREATE TABLE AS SELECT`** statement. The following statement works as follows:
-- MAGIC * Creates a table named *trips_external* in the *default* schema of the *main* catalog
-- MAGIC * Uses the *trips* table from the *nyctaxi* schema of the *samples* catalog as a model for the table schema and data
-- MAGIC * Locates the data files backing the table in the *trips_external* folder of the external storage bucket using the **`LOCATION`** keyword
-- MAGIC
-- MAGIC Note that we would use a similar statement if the storage container already contained data files that we wanted to build a table around.

-- COMMAND ----------

CREATE TABLE main.default.trips_external
  LOCATION 's3://dbacademy-test-external-bucket/trips_external'
  AS SELECT * FROM samples.nyctaxi.trips

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Let's examine the table properties. We'll see this is a Delta table with data files residing on the external storage bucket we created earlier.

-- COMMAND ----------

DESCRIBE TABLE EXTENDED main.default.trips_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Dropping an external table is the same as dropping a conventional managed table, though there is one subtle difference that we'll see shortly.

-- COMMAND ----------

DROP TABLE main.default.trips_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Accessing files
-- MAGIC
-- MAGIC Let's do some file access; let's read the contents of the topmost folder in the container.

-- COMMAND ----------

LIST 's3://dbacademy-test-external-bucket'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Here we see the only item so far is the *trips_external* folder. Let's examine the contents of that.

-- COMMAND ----------

LIST 's3://dbacademy-test-external-bucket/trips_external'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Notice that, even though we dropped the table, the data files remain. This is intrinsic behaviour of an external table, which is the main differentiator between managed and external tables. When we drop a managed table, the data files are deleted as well. As it stands, you could recreate the table if desired, without having to specify source data.

-- COMMAND ----------

CREATE TABLE main.default.trips_external LOCATION 's3://dbacademy-test-external-bucket/trips_external'

-- COMMAND ----------

SELECT * FROM main.default.trips_external

-- COMMAND ----------

-- MAGIC %md
-- MAGIC If you want to purge the data files you must do this outside the scope of the Databricks platform.

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
