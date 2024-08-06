// Databricks notebook source
// MAGIC %md-sandbox
// MAGIC
// MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
// MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
// MAGIC </div>

// COMMAND ----------

// MAGIC %md
// MAGIC # Integrating Kinesis
// MAGIC
// MAGIC In this demo you will learn how to:
// MAGIC * Set up a sample data stream using Amazon Kinesis
// MAGIC * Connect Databricks to Kinesis
// MAGIC * Ingest records from a Kinesis stream and store them in a Delta table

// COMMAND ----------

// MAGIC %md
// MAGIC ## Prerequisites
// MAGIC
// MAGIC If you would like to follow along with this demo, you will need:
// MAGIC * Administrator access to your AWS console, with the ability to create buckets and IAM roles
// MAGIC * Account administrator capabilities in your Databricks account

// COMMAND ----------

// MAGIC %md
// MAGIC ## Overview
// MAGIC
// MAGIC The world operates in real time. Therefore, it follows that your organization will need to make business decisions in real time, whether those decisions are based on sound analytics or not. Spark structured streaming, tightly integrated with Databricks, unlocks the ability to integrate both batch and near real-time processing in your ETL pipelines without having to learn new tools or APIs.
// MAGIC
// MAGIC AWS Kinesis is a massively scalable and durable real-time data streaming service managed natively by AWS. With the ability to continuously capture gigabytes of data per second from hundreds of thousands of sources, this robust messaging service can be used in a variety of applications including:
// MAGIC
// MAGIC * Analytics pipelines like clickstreams
// MAGIC * Anomaly detection (fraud/outliers)
// MAGIC * Application logging
// MAGIC * Archiving data
// MAGIC * Data collection from IoT devices
// MAGIC * Device and user telemetry streaming and processing
// MAGIC * Transaction processing
// MAGIC * Live dashboarding
// MAGIC
// MAGIC In this lab, we'll walk through how to integrate Kinesis into your Databricks-powered applications so that you can take advantage of this powerful streaming service.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Setting up a sample data stream
// MAGIC
// MAGIC In order demonstrate connecting your application to Kinesis, we must have a target data stream. If your organization doesn't already have one set up for use, we will create one in this section.
// MAGIC
// MAGIC 1. In AWS Kinesis, let's choose a region. For this example, let's choose *us-east-1*.
// MAGIC 1. Let's create a Kinesis data stream. We begin the process by clicking **Create data stream**. Note that the layout of this page may be different depending on whether or not you already have a data stream created in the selected region.
// MAGIC    1. Specify a **Data stream name**. Let's use *dbacademy-test-data-stream*.
// MAGIC    1. For this example, let's avoid automatic scaling by selecting **Provisioned** and setting **Provisioned shards** to *1*.
// MAGIC    1. Click **Create data stream**.
// MAGIC 1. Note the ARN of the newly created data stream, as we will need this momentarily.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Authentication
// MAGIC
// MAGIC Databricks uses Amazon’s <a href="https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/credentials.html#credentials-default" target="_blank">default credential provider chain</a> by default. This provides for a number of authentication options; for example:
// MAGIC
// MAGIC * Explicitly specifying credentials when establishing the connection
// MAGIC * Specifying credentials using environment variables or Java system properties
// MAGIC * A shared AWS credentials file
// MAGIC * Using instance profiles to allow cluster instances to assume an IAM role
// MAGIC
// MAGIC Databricks recommends the latter; launching your clusters with an instance profile that has permissions to access your Kinesis data stream. This is the approach we'll cover in this lab.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Creating an IAM role
// MAGIC
// MAGIC An instance profile is simply an encapsulation of an IAM role that can be associated with EC2 instances, the virtual machines that make up Databricks clusters. Let's get started on this path by creating an IAM role with a policy that enables us to access the data stream created earlier.
// MAGIC
// MAGIC 1. In the AWS IAM console, let's select **Roles**.
// MAGIC 1. Click **Create role**.
// MAGIC 1. Choose **AWS service**. This will let us set up a role that can access a service (in this case, Kinesis).
// MAGIC    * Now let's select **EC2** as the use case.
// MAGIC    * Let's click **Next** until we get to the final page.
// MAGIC    * Let's assign the name for our role (use *dbacademy-test-kinesis-access-role*).
// MAGIC    * Click **Create role**.
// MAGIC 1. Now let's locate the role we just created and select it.
// MAGIC 1. Let's click the **Permissions** tab, then select **Add permissions > Create inline policy**.
// MAGIC 1. Click the **JSON** tab.
// MAGIC 1. Replace the default policy with the following:
// MAGIC     ```
// MAGIC     {
// MAGIC         "Version": "2012-10-17",
// MAGIC         "Statement": [
// MAGIC             {
// MAGIC                 "Sid": "Stmt123",
// MAGIC                 "Effect": "Allow",
// MAGIC                 "Action": [
// MAGIC                     "kinesis:DescribeStream",
// MAGIC                     "kinesis:PutRecord",
// MAGIC                     "kinesis:PutRecords",
// MAGIC                     "kinesis:GetShardIterator",
// MAGIC                     "kinesis:GetRecords",
// MAGIC                     "kinesis:ListShards",
// MAGIC                     "kinesis:DescribeStreamSummary",
// MAGIC                     "kinesis:RegisterStreamConsumer"
// MAGIC                 ],
// MAGIC                 "Resource": [
// MAGIC                     "<KINESIS_ARN>"
// MAGIC                 ]
// MAGIC             },
// MAGIC             {
// MAGIC                 "Sid": "Stmt234",
// MAGIC                 "Effect": "Allow",
// MAGIC                 "Action": [
// MAGIC                     "kinesis:SubscribeToShard",
// MAGIC                     "kinesis:DescribeStreamConsumer"
// MAGIC                 ],
// MAGIC                 "Resource": [
// MAGIC                     "<KINESIS_ARN>/*"
// MAGIC                 ]
// MAGIC             }
// MAGIC         ]
// MAGIC     }
// MAGIC     ```
// MAGIC 1. Before we proceed, let's replace the instances of *&lt;KINESIS_ARN&gt;* with the ARN of the data stream we just created.
// MAGIC 1. Now let's click **Review policy** to get to the final page.
// MAGIC 1. Let's assign the name for our policy (use *dbacademy-test-policy-kinesis*).
// MAGIC 1. Click **Create policy**.
// MAGIC 1. Let's take note of the **ARN** and **Instance profile ARN**; we will need these soon.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Updating the EC2 policy
// MAGIC
// MAGIC In order to attach the created instance profile to EC2 instances (that is, the virtual machines making up Databricks clusters), we must add the ability to pass the role in the cross-account IAM role that allows the Databricks control plane to start and manage clusters. Let's do that now. 
// MAGIC
// MAGIC 1. Still in **IAM > Roles**, let's locate the appropriate role. If you followed the labs from *AWS Databricks Platform Administration Fundamentals*, look for the role named *dbacademy-test-cross-account-role*.
// MAGIC 1. Click the **Permissions** tab and select the policy.
// MAGIC 1. In the **JSON** tab, add the following statement to the policy, replacing **`<ROLE_ARN>`** with the ARN of the IAM role created earlier:
// MAGIC     ```
// MAGIC     {
// MAGIC       "Effect": "Allow",
// MAGIC       "Action": "iam:PassRole",
// MAGIC       "Resource": "<ROLE_ARN>"
// MAGIC     }
// MAGIC     ```
// MAGIC 1. Now let's click **Review policy** to get to the final page.
// MAGIC 1. Let's click **Save changes**.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Creating an instance profile
// MAGIC
// MAGIC With a backing IAM role created and configured to access the Kinesis data stream, let's add the instance profile to Databricks.
// MAGIC
// MAGIC 1. Let's log in to a deployed workspace as a user with administrative privileges.
// MAGIC 1. Go to the **Admin Console**.
// MAGIC 1. Click the **Instance profiles** tab.
// MAGIC 1. Click **Add instance profile**.
// MAGIC 1. Paste the instance profile ARN of the IAM role created earlier; Databricks will automatically validate the ARN and populate the IAM role ARN.
// MAGIC 1. Finally, let's click **Add**.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Applying the instance profile
// MAGIC
// MAGIC An instance profile applies to Databricks clusters, so in order to make use of one, we must attach the instance profile to an existing cluster or create a new one. Let's create one now.
// MAGIC 1. Click **Connect > Create new resource...** from any notebook then select **Advanced Configuration**. Alternatively, go to the **Compute** page of the Data Science and Engineering workspace and click **Create compute**.
// MAGIC 1. Configure the cluster, paying particular attention to the following settings:
// MAGIC     * Set **Access mode** to *Single user* (to support the demo application which is implemented in Scala)
// MAGIC     * Set **Single user access** to the user who will be running the streaming application to follow
// MAGIC     * Select the **Instance profile** created in the previous section.
// MAGIC 1. Attach the newly created cluster to this notebook and proceed to the next section.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Implementing a simple streaming application
// MAGIC
// MAGIC In this section, we will implement a simple streaming application that ingests data from Kinesis, and writes it to a Delta table.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Implementing the consumer
// MAGIC
// MAGIC The following code (adapted from the Kinesis WordCount example in the <a href="https://docs.databricks.com/structured-streaming/kinesis.html" target="_blank">Databricks documentation</a>) implements a simple application that ingests data from a Kinesis data stream, writing the output to a Delta table named *kinesis_demo_bronze* in the *default* schema of the *main* catalog. If needed, subsitute values for *kinesisStreamName* and *kinesisRegion* if you used different values for your data stream.
// MAGIC
// MAGIC Ensuring that the notebook is attached to the cluster configured with the instance profile, let's run the following cell. Since it's a streaming read/write, it will run indefinitely. Let's allow it to run, and let's open the stream viewer to see record activity. As there is no data posted to the data stream, no data will be streamed yet.

// COMMAND ----------

import com.amazonaws.services.kinesis.model.PutRecordRequest
import com.amazonaws.services.kinesis.AmazonKinesisClientBuilder
import com.amazonaws.auth.{AWSStaticCredentialsProvider, BasicAWSCredentials}
import java.nio.ByteBuffer
import scala.util.Random

val kinesisStreamName = "dbacademy-test-data-stream"
val kinesisRegion = "us-east-1"

val kinesis = spark.readStream
  .format("kinesis")
  .option("streamName", kinesisStreamName)
  .option("region", kinesisRegion)
  .option("initialPosition", "TRIM_HORIZON")
  .load()
  .writeStream
  .format("delta")
  .option("checkpointLocation", "/tmp/kinesis-demo/_checkpoint")
  .table("main.default.kinesis_demo_bronze")

// COMMAND ----------

// MAGIC %md
// MAGIC ### Implementing the producer
// MAGIC
// MAGIC Let's now implement a simple producer that posts records to the demo data stream. Run the following cell, then observe the effects in the view in the previous section.

// COMMAND ----------

val kinesisClient = AmazonKinesisClientBuilder.standard()
  .withRegion(kinesisRegion)
  .build()

println(s"Putting words onto stream $kinesisStreamName")
var lastSequenceNumber: String = null

for (i <- 0 to 10) {
  val time = System.currentTimeMillis
  for (word <- Seq("Through", "three", "cheese", "trees", "three", "free", "fleas", "flew", "While", "these", "fleas", "flew", "freezy", "breeze", "blew", "Freezy", "breeze", "made", "these", "three", "trees", "freeze", "Freezy", "trees", "made", "these", "trees", "cheese", "freeze", "That's", "what", "made", "these", "three", "free", "fleas", "sneeze")) {
    val data = s"$word"
    val partitionKey = s"$word"
    val request = new PutRecordRequest()
        .withStreamName(kinesisStreamName)
        .withPartitionKey(partitionKey)
        .withData(ByteBuffer.wrap(data.getBytes()))
    if (lastSequenceNumber != null) {
      request.setSequenceNumberForOrdering(lastSequenceNumber)
    }    
    val result = kinesisClient.putRecord(request)
    lastSequenceNumber = result.getSequenceNumber()
  }
  Thread.sleep(math.max(10000 - (System.currentTimeMillis - time), 0)) // loop around every ~10 seconds 
}

// COMMAND ----------

// MAGIC %md
// MAGIC ## Conclusion and clean-up
// MAGIC
// MAGIC As the producer posts data to the Kinesis data stream, you'll notice that the processed records are reflected in the consumer cell that performs a streaming ingestion and write into the Delta table. Feel free to use the data explorer to view the destination table, *main.default.kinesis_demo_bronze*. The key take-away here is not the content of the application or the data it synthesizes, however; the main point relates to the authentication scheme we set up to enable Databricks workloads to seamlessly access Kinesis data streams through the use of instance profiles/IAM roles. While there are other options for authenticating with this service, Databricks considers this approach to be the most secure and thus endorses it as a best practice.
// MAGIC
// MAGIC When done, please be sure to cancel the consumer and producer (if still running). This will allow your cluster to idle (and automatically terminate, if it remains idle for a sufficiently long period of time).

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
// MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
// MAGIC <br/>
// MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
