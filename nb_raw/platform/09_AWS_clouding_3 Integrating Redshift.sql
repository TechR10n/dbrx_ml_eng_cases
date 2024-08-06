-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Integrating Redshift
-- MAGIC
-- MAGIC In this demo you will learn how to:
-- MAGIC * Set up a sample Redshift data warehouse
-- MAGIC * Connect Databricks to Redshift
-- MAGIC * Read data from Redshift, process it, and write it back

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Prerequisites
-- MAGIC
-- MAGIC If you would like to follow along with this demo, you will need:
-- MAGIC * Administrator access to your AWS console, with the ability to create buckets, IAM roles, and Redshift clusters
-- MAGIC * Account administrator capabilities in your Databricks account

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Overview
-- MAGIC
-- MAGIC Amazon Redshift is a high-performance data warehouse that excels at providing high-concurrency data access. Amazon Redshift can query data from a Databricks Lakehouse, and analytics-ready data can be ingested into Redshift for reporting and querying by BI platforms.
-- MAGIC
-- MAGIC In this lab, we'll walk through how to integrate Redshift into your Databricks-powered applications so that you can take advantage of this powerful data warehouse service.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Setting up a sample warehouse
-- MAGIC
-- MAGIC In order demonstrate connecting your application to Redshift, we must have a target Redshift cluster. If your organization doesn't already have one set up for use, we will create one in this section.
-- MAGIC
-- MAGIC 1. In AWS Redshift, choose a region. This does not have to coincide with the workspace, but for this example let's choose *us-east-1*.
-- MAGIC 1. Let's click **Create cluster**.
-- MAGIC    1. Specify a **Cluster identifier**. Let's use *dbacademy-test-redshift-cluster*.
-- MAGIC    1. To minimize cost for this example, specify *dc2.large* and *1* for **Node type** and **Number of nodes**, respectively.
-- MAGIC    1. Let's take advantage of sample data by enabling **Load sample data**.
-- MAGIC    1. Specify *awsuser* and *AWSuser1* for **Admin user name** and **Admin user password**, respectively.
-- MAGIC    Click **Create cluster**.
-- MAGIC    
-- MAGIC The cluster will take a few minutes to be created. Once it's running, we can test it out.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Querying from Redshift
-- MAGIC
-- MAGIC Let's try running a query against our new Redshift cluster using the Redshift query editor.
-- MAGIC
-- MAGIC 1. After the status of our cluster transitions to *Available*, let's click **Query data**.
-- MAGIC 1. Expand the element representing the cluster. Now let's expand **sample_data_dev > tickit > Tables**, creating the database on the way if prompted.
-- MAGIC 1. Double-click **users**. A simple query appears in a new tab.
-- MAGIC 1. Let's run the query. The results appear below, limited to 100 rows.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating the tempdir bucket
-- MAGIC
-- MAGIC S3 acts as an intermediary to store bulk data when reading from or writing to Redshift. Because the data source does not clean up the temporary files that it creates in S3, Databricks recommends using a dedicated temporary S3 bucket with an object lifecycle configuration to ensure that temporary files are automatically deleted after a specified expiration period. Let's create and configure a bucket like this now.
-- MAGIC
-- MAGIC 1. In AWS S3, click **Create bucket**.
-- MAGIC 1. Let's specify a name. When choosing your own names, be mindful to not include dots in your names. Bucket names must also be globally unique. In this example we use *dbacademy-test-redshift-bucket*, but you should include a suffix or prefix that uniquely ties the name to your organization; for example, replace *dbacademy* with your domain name (using hyphens instead of dots).
-- MAGIC 1. Let's choose a region; note that this does not have to coincide with the workspace or Redshift, but if the bucket resides in a different region than Redshift, some <a href="https://docs.databricks.com/external-data/amazon-redshift.html#s3-bucket-and-redshift-cluster-are-in-different-aws-regions" target="_blank">additional configuration</a> is needed. To keep things less complex for now, let's choose the same regions as we did for our Redshift cluster.
-- MAGIC 1. For this example, let's accept the default settings for the rest, and create the bucket.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Configuring object lifecycle management
-- MAGIC
-- MAGIC The Redshift data source uses the S3 bucket extensively for temporal storage, but it doesn't clean up the files it creates. To avoid pileup of stale data and cost, we will rely on the object lifecycle management that S3 provides to do this for us. In this section we'll configure a lifecycle rule to expire objects after a predefined amount of time.
-- MAGIC
-- MAGIC 1. Let's locate and select the bucket we just created.
-- MAGIC 1. In the **Management** tab, let's click **Create lifecycle rule**.
-- MAGIC 1. Specify a name for the rule (let's use *dbacademy-test-redshift-expire-objects*). Note that these names only have to be unique for the associated bucket.
-- MAGIC 1. Let's select **Apply to all objects in the bucket**, and also check the box to confirm that the rule will be applied to all objects in the bucket.
-- MAGIC 1. Select the **Expire current versions of objects** action. Since our bucket does not have version control enabled, the object expiration that results from executing this action will cause the object to be deleted automatically by S3.
-- MAGIC 1. Let's specify *1* for **Days after object creation**.
-- MAGIC 1. Finally, let's click **Create rule**.
-- MAGIC
-- MAGIC Note that this rule only imposes expirations on objects in the bucket. The actual deletion happens asynchronously. As a result, objects typically persist for some time after they expire, though you are not charged for storing objects that have expired.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuring Spark authentication to S3
-- MAGIC
-- MAGIC In order to copy data through to or from Redshift, your Databricks clusters (provided by EC2) must be authorized to access S3. Databricks generally advocates using instance profiles (that is, IAM roles) to authorize this connnection. Let's create an instance profile that will enable our cluster to access the bucket.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating the IAM role
-- MAGIC
-- MAGIC Let's first create the IAM role that will back the instance profile.
-- MAGIC
-- MAGIC 1. In the AWS IAM console, let's select **Roles**.
-- MAGIC 1. Click **Create role**.
-- MAGIC 1. Choose **AWS service**. This will let us set up a role that can be used by a service (in this case, EC2).
-- MAGIC    * Now let's select **EC2** as the use case.
-- MAGIC    * Let's click **Next** until we get to the final page.
-- MAGIC    * Let's assign the name for our role (use *dbacademy-test-redshift-access-role*).
-- MAGIC    * Click **Create role**.
-- MAGIC 1. Now let's add an inline policy to permit S3 read/write operations.
-- MAGIC    * Let's locate the role we just created and select it.
-- MAGIC    * In the **Permissions** tab, select **Add permissions > Create inline policy**.
-- MAGIC    * In the **JSON** tab, replace the default policy with the following. Replace both instances of **`<BUCKET>`** with the name of bucket you created.
-- MAGIC     ```
-- MAGIC     {
-- MAGIC       "Version": "2012-10-17",
-- MAGIC       "Statement": [
-- MAGIC         {
-- MAGIC           "Effect": "Allow",
-- MAGIC           "Action": [
-- MAGIC             "s3:ListBucket"
-- MAGIC           ],
-- MAGIC           "Resource": [
-- MAGIC             "arn:aws:s3:::<BUCKET>"
-- MAGIC           ]
-- MAGIC         },
-- MAGIC         {
-- MAGIC           "Effect": "Allow",
-- MAGIC           "Action": [
-- MAGIC             "s3:PutObject",
-- MAGIC             "s3:GetObject",
-- MAGIC             "s3:DeleteObject",
-- MAGIC             "s3:PutObjectAcl"
-- MAGIC           ],
-- MAGIC           "Resource": [
-- MAGIC             "arn:aws:s3:::<BUCKET>/*"
-- MAGIC           ]
-- MAGIC         }
-- MAGIC       ]
-- MAGIC     }
-- MAGIC     ```
-- MAGIC    * Now let's click **Review policy** to get to the final page.
-- MAGIC    * Let's assign the name for our policy (use *dbacademy-test-redshift-access-policy*).
-- MAGIC    * Click **Create policy**.
-- MAGIC 1. Let's take note of the **ARN** and **Instance profile ARN**; we will need these soon.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Updating the EC2 policy
-- MAGIC
-- MAGIC In order to attach the created instance profile to EC2 instances (that is, the virtual machines making up Databricks clusters), we must add the ability to pass the role in the cross-account IAM role that allows the Databricks control plane to start and manage clusters. Let's do that now. 
-- MAGIC
-- MAGIC 1. Still in **IAM > Roles**, let's locate the appropriate role. If you followed the labs from *AWS Databricks Platform Administration Fundamentals*, look for the role named *dbacademy-test-cross-account-role*.
-- MAGIC 1. In the **Permissions** tab, select the policy.
-- MAGIC 1. In the **JSON** tab, add the following statement to the policy, replacing **`<ROLE_ARN>`** with the ARN of the IAM role created earlier:
-- MAGIC     ```
-- MAGIC     {
-- MAGIC       "Effect": "Allow",
-- MAGIC       "Action": "iam:PassRole",
-- MAGIC       "Resource": "<ROLE_ARN>"
-- MAGIC     }
-- MAGIC     ```
-- MAGIC 1. Now let's click **Review policy** to get to the final page.
-- MAGIC 1. Let's click **Save changes**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating an instance profile
-- MAGIC
-- MAGIC With a backing IAM role created and configured to access S3, let's add the instance profile to Databricks.
-- MAGIC
-- MAGIC 1. Let's log in to a deployed workspace as a user with administrative privileges.
-- MAGIC 1. Go to the **Admin Console**.
-- MAGIC 1. Click the **Instance profiles** tab.
-- MAGIC 1. Click **Add instance profile**.
-- MAGIC 1. Paste the instance profile ARN of the IAM role created earlier; Databricks will automatically validate the ARN and populate the IAM role ARN.
-- MAGIC 1. Finally, let's click **Add**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Applying the instance profile
-- MAGIC
-- MAGIC An instance profile applies to Databricks clusters, so in order to make use of one, we must attach the instance profile to an existing cluster or create a new one. Let's create one now.
-- MAGIC 1. Click **Connect > Create new resource...** from any notebook then select **Advanced Configuration**. Alternatively, go to the **Compute** page of the Data Science and Engineering workspace and click **Create compute**.
-- MAGIC 1. Configure the cluster, paying particular attention to the following settings:
-- MAGIC     * Set **Access mode** to *Single user*
-- MAGIC     * Set **Single user access** to the user who will be running the application to follow
-- MAGIC     * Select the **Instance profile** created in the previous section.
-- MAGIC 1. Attach the newly created cluster to this notebook and proceed to the next section.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Testing the connection
-- MAGIC
-- MAGIC Let's test the connection. Replace **`<BUCKET>`** with your bucket name. Attach this notebook to the cluster we just created (or edited) and run the following cell. This generates a simple DataFrame and writes it to the *test_spark* folder of the tempdir bucket.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC (spark.range(10)
-- MAGIC     .write
-- MAGIC     .format("parquet")
-- MAGIC     .mode("overwrite")
-- MAGIC     .save("s3a://<BUCKET>/test_spark")
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now let's test the ability to read by reading back the data files we just created. Once again, replace **`<BUCKET>`** with your bucket name before running the cell.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(spark.read.format("parquet").load("s3a://<BUCKET>/test_spark"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Furthermore, if we revisit the bucket in the S3 console, we will see the presence of the *test_spark* folder. We don't have to clean this up though, since the object lifecycle rule we set up will take care of this for us.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuring Redshift authentication to S3
-- MAGIC
-- MAGIC We have configured the setup so that EC2 can access S3, but Redshift must be authorized as well. In this section we will work through authorizing this connection.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Modifying the IAM role
-- MAGIC
-- MAGIC Rather than complicating things by creating an additional IAM role, let's augment the one we created in the previous section authorizing EC2. We will modify its trust policy to include the Redshift service in addition to EC2.
-- MAGIC
-- MAGIC 1. In **IAM > Roles**, let's locate the role we created (*dbacademy-test-redshift-access-role*).
-- MAGIC 1. In the **Trust relationships** tab, let's click **Edit trust policy**.
-- MAGIC 1. Click the **Add** button to add a principal.
-- MAGIC 1. Select **AWS services** for the **Principal type**.
-- MAGIC 1. Change the **ARN** field to read *redshift.amazonaws.com*.
-- MAGIC 1. Click **Add principal**.
-- MAGIC 1. Finally, let's click **Update policy**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Associating the IAM role with Redshift
-- MAGIC
-- MAGIC Just like we had to associate an instance profile with the Databricks cluster, so too must we associate the IAM role with the Redshift cluster in order for Redshift to access the tempdir bucket.
-- MAGIC
-- MAGIC 1. In **Redshift > Clusters**, let's locate and select the cluster we created (*dbacademy-test-redshift-cluster*).
-- MAGIC 1. Select **Actions > Manage IAM roles**.
-- MAGIC 1. Select the role we created (*dbacademy-test-redshift-access-role*).
-- MAGIC 1. Now let's click **Associate IAM role**, then click **Save changes**.
-- MAGIC
-- MAGIC The cluster will be out of service for a short while. Once the status transitions back to *Available*, we may proceed.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Testing the connection
-- MAGIC
-- MAGIC Let's test the ability for Redshift to reach the S3 bucket using an **`UNLOAD`** operation, which will copy out a table from Redshift to data files in S3.
-- MAGIC
-- MAGIC Returning to the Redshift query editor, let's wrap the previous query so that it appears as follows. Replace **`<BUCKET>`** with the bucket name and **`<ROLE_ARN>`** with the ARN of the IAM role we created (*dbacademy-test-redshift-access-role*).
-- MAGIC    ```
-- MAGIC    UNLOAD ('SELECT * FROM "sample_data_dev"."tickit"."users"')
-- MAGIC      TO 's3://<BUCKET>/test_redshift/'
-- MAGIC      CREDENTIALS 'aws_iam_role=<ROLE_ARN>' 
-- MAGIC    ```
-- MAGIC
-- MAGIC After running the query, let's revisit the bucket in the S3 console. Assuming the query succeeded, we will see the presence of the *test_redshift* folder. We don't have to clean this up though, since the object lifecycle rule we set up will take care of this for us.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Connecting EC2 to Redshift
-- MAGIC
-- MAGIC From an authentication point of view, the setup is complete. But before this entire thing can work, we must give consideration to one last thing, which is the IP connectivity between our Databricks clusters and the Redshift cluster. Both these things run in separate VPCs, and indeed it should be this way. Without additional configuration, Redshift uses the default VPC in the account and Databricks will create its own dedicated VPC. But this means that these two will not have mutual visibility.
-- MAGIC
-- MAGIC We can demonstrate this by executing the following cell. This command utlizes the **`nc`** utility (netcat) to establish a connection to the Redshift endpoint on its designated port, 5439. Before attempting to run this, you will need to replace **`<ENDPOINT_DNS>`** with the domain name component of the Redshift endpoint value.

-- COMMAND ----------

-- MAGIC %sh nc -w 10 -zv <ENDPOINT_DNS> 5439

-- COMMAND ----------

-- MAGIC %md
-- MAGIC This command either times out or fails to resolve. In this section, we'll use VPC peering to fix this, which provides a secure way for these two services to reach each other.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Peering VPCs
-- MAGIC While configuring VPC peering is fairly straightforward, there are a number of parameters to keep track of in order to configure things properly. To assist, we present this table which you can fill in on your own as we gather the needed bits of information to complete this configuration.
-- MAGIC
-- MAGIC | AWS Service        | Description                 | ID | CIDR |
-- MAGIC | ------------------ | --------------------------- | -- | ---- |
-- MAGIC | VPC                | Databricks VPC              |    |      |
-- MAGIC | VPC                | Redshift VPC                |    |      |
-- MAGIC | Route table        | Databricks VPC              |    |      |
-- MAGIC | Peering connection | Databricks &#8596; Redshift |    |      |
-- MAGIC | Security group     | Databricks unmanaged        |    |      |
-- MAGIC
-- MAGIC 1. Locate the VPCs for your Databricks workspace and Redshift. Record the ID and CIDR of both.
-- MAGIC 1. Let's create the peering connection.
-- MAGIC    * Open the VPC dashboard and select **Peering connections**.
-- MAGIC    * Click **Create peering connection**.
-- MAGIC    * Specify a name (let's use *dbacademy-test-vpc-peering-connection*).
-- MAGIC    * Set the **VPC ID (Requester)** to the Databricks VPC ID.
-- MAGIC    * Select **My account** and **This region**.
-- MAGIC    * Set the **VPC ID (Acceptor)** to the Redshift VPC ID.
-- MAGIC    * Click **Create peering connection**. Let's record the ID of the peering connection in the table.
-- MAGIC    * The connection request must be accepted. Let's accept it by selecting **Actions > Accept request**, then clicking **Accept request**.
-- MAGIC 1. Now let's enable DNS resolution in both VPCs.
-- MAGIC    * Select **Actions > Edit DNS settings**.
-- MAGIC    * Enable both **Requester DNS resolution** and **Accepter DNS resolution**.
-- MAGIC    * Click **Save changes**.   
-- MAGIC 1. Now let's add a routing to the Databricks VPC.
-- MAGIC    * Back in the VPC dashboard, select **Route tables**.
-- MAGIC    * Identify the entries associated with the Databricks VPC. Select the main routing table.
-- MAGIC    * In the **Routes** tab, click **Edit routes**.
-- MAGIC    * Let's click **Add route**.
-- MAGIC    * Enter the Redshift CIDR and peering connection ID for **Destination** and **Target**, respectively.
-- MAGIC    * Click **Save changes**.
-- MAGIC 1. Now let's add a complementary routing to the Redshift VPC.
-- MAGIC    * Back in the route tables, identify the entries associated with the Redshift VPC. Select the main routing table.
-- MAGIC    * In the **Routes** tab, click **Edit routes**.
-- MAGIC    * Let's click **Add route**.
-- MAGIC    * Enter the Databricks CIDR and peering connection ID for **Destination** and **Target**, respectively.
-- MAGIC    * Click **Save changes**.
-- MAGIC 1. Finally, let's add a new rule to the the Redshift security group to allow inbound connections. 
-- MAGIC    * In the VPC dashboard, select **Security groups**.
-- MAGIC    * Locate the security group associated with the Databricks VPC with *unmanaged* in its name. Record its ID.
-- MAGIC    * Locate and select the security group associated with your Redshift cluster. You can obtain this from the **Network and security settings** for your Redshift cluster.
-- MAGIC    * In the **Inbound rules** tab, click **Edit inbound rules**.
-- MAGIC    * Let's click **Add rule**.
-- MAGIC    * Specify *Redshift* for **Type**. For **Source**, select **Custom**, then enter the unmanaged security group ID.
-- MAGIC    * Click **Save rules**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Testing the connection
-- MAGIC
-- MAGIC Let's once again test the connection using the **`nc`** command, again replacing **`<ENDPOINT_DNS>`** with the domain name component of the Redshift endpoint value. 

-- COMMAND ----------

-- MAGIC %sh nc -w 10 -zv <ENDPOINT_DNS> 5439

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Assuming the configuration was done properly, the command should succeed almost immediately. We're now ready to begin working with Redshift from our Databricks cluster.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Interacting with Redshift
-- MAGIC
-- MAGIC With all the connections in place, reading data from Spark is a relatively straightforward process using the *redshift* source and configuring options appropriately. Most importantly:
-- MAGIC * *search_path* specifies the schema
-- MAGIC * *dbtable* specifies the table
-- MAGIC * *tempdir* specifies full path within the tempdir bucket for exchanging data
-- MAGIC * *url* specifies the JDBC URL for the Redshift endpoint
-- MAGIC * *user* and *password* specify the JDBC credentials for authenticating with Redshift. Currently this is the only way to authenticate this connection.
-- MAGIC * *aws_iam_role* specifies the credential to use for accessing S3; that is, the ARN of the IAM role

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Reading from Redshift
-- MAGIC
-- MAGIC Let's query the *users* table from Spark as we did previously in the Redshift query editor.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = (spark.read
-- MAGIC   .format("redshift")
-- MAGIC   .option("search_path", "tickit")
-- MAGIC   .option("dbtable", "users")
-- MAGIC   .option("tempdir", "s3a://<BUCKET>/tmp")
-- MAGIC   .option("url", "<JDBC_URL>")
-- MAGIC   .option("user", "awsuser")
-- MAGIC   .option("password", "AWSuser1")
-- MAGIC   .option("aws_iam_role", "<ROLE_ARN>")
-- MAGIC   .load()
-- MAGIC )
-- MAGIC
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The above can equivalently be done in SQL, though we need to locally create an intermediate table since we can't specify options when selecting directly from a source.

-- COMMAND ----------

DROP TABLE IF EXISTS redshift_table;
CREATE TABLE redshift_table
  USING redshift
  OPTIONS(
    search_path 'tickit',
    dbtable 'users',
    tempdir 's3a://<BUCKET>/tmp',
    url '<JDBC_URL>',
    user 'awsuser',
    password 'AWSuser1',
    aws_iam_role '<ROLE_ARN>'
  );

-- COMMAND ----------

SELECT * FROM redshift_table;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Writing to Redshift
-- MAGIC
-- MAGIC Now let's perform some minor transformations; let's redact the *email* and *phone* columns and write the results back to a similarly named table in the *public* schema. Ensure that you perform the appropriate substitutions prior to running the cell.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import lit
-- MAGIC
-- MAGIC df_transformed = df.withColumn("email",lit("REDACTED")).withColumn("phone",lit("REDACTED"))
-- MAGIC
-- MAGIC (df_transformed.write
-- MAGIC   .format("redshift")
-- MAGIC   .option("search_path", "public")
-- MAGIC   .option("dbtable", "users")
-- MAGIC   .option("tempdir", "s3a://<BUCKET>/tmp")
-- MAGIC   .option("url", "<JDBC_URL>")
-- MAGIC   .option("user", "awsuser")
-- MAGIC   .option("password", "AWSuser1")
-- MAGIC   .option("aws_iam_role", "<ROLE_ARN>")
-- MAGIC   .mode("overwrite")
-- MAGIC   .save()
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
