-- Databricks notebook source
-- INCLUDE_HEADER_TRUE
-- INCLUDE_FOOTER_TRUE

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Managing interactive clusters
-- MAGIC
-- MAGIC In this lab, you will learn how to:
-- MAGIC * Implement some common schemes to control compute costs associated with interactive clusters
-- MAGIC * Pre-create clusters for users
-- MAGIC * Control access to clusters
-- MAGIC * Employ a cluster policy to constrain costs and impose consistency

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Prerequisites
-- MAGIC
-- MAGIC If you would like to follow along with this lab, you will need workspace administrator capabilities.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Overview
-- MAGIC
-- MAGIC Interactive (all-purpose) clusters represent a family of compute resources that your users will rely on heavily, particularly early in the development stage. Many options are available, and there can be significant cost associated with some configurations. Managing compute resources therefore represents a trade-off between convenience and cost control. Management strategies can be quantized into three broad approaches:
-- MAGIC * Unrestricted cluster creation
-- MAGIC * Disallow cluster creation
-- MAGIC * Restrict cluster creation
-- MAGIC
-- MAGIC There are pros and cons to each, which we will cover in this lab.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Unrestricted cluster creation
-- MAGIC
-- MAGIC In this "free-for-all" scenario, all users are granted the unrestricted capability to create their own clusters when needed.
-- MAGIC
-- MAGIC The main benefit of this approach is that it's trivial to implement; a simple entitlement grant to all users is all that's needed; and there is no adminsterial effort or potential bottlenecks required to support it. Whenever users need compute resources, they have the ability to serve themselves.
-- MAGIC
-- MAGIC The main downside is the potentially unbounded cost associated this approach. Without discipline on the part of your users, compute costs could be high. This approach also complicates housekeeping. Suppose, for example, you need to tag clusters to track costs. Under this cluster management approach, this practice is difficult to enforce.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Enabling unrestricted cluster creation
-- MAGIC
-- MAGIC Let's set up an environment where all users can create clusters in an unrestricted manner.
-- MAGIC 1. In the admin console, click the **Groups** tab.
-- MAGIC 1. Select the *users* group, a special workspace local group capturing all users.
-- MAGIC 1. Click the **Entitlements** tab.
-- MAGIC 1. Enable **Allow unrestricted cluster creation**.
-- MAGIC
-- MAGIC With this entitlement in place, all users (including all future users that get assigned to this workspace) will be able to create compute resources with no restrictions.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Disabled cluster creation
-- MAGIC
-- MAGIC This scenario represents the complete opposite of the previous one. No users (with the exception of workspace administrators) are permitted to create clusters. Rather, administrators create clusters on behalf of the users, either on demand or by anticipating needs and pre-populating the workspace.
-- MAGIC
-- MAGIC Since this approach is the opposite of the previous, the pros and cons are essentially reversed. The main benefit of this approach is that it delivers predictable cost; resources can be allocated with a budget in mind, and the organization is essentially guaranteed not to exceed that budget. Administrators can also ensure consistency and tagging to help track costs.
-- MAGIC
-- MAGIC The downside to this approach is that it's a process that requires upfront and/or ongoing effort on the part of the workspace administrators to support, and it introduces the potential for bottlenecks and frustration if users can't get the resources they need when they need them.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Preventing user cluster creation
-- MAGIC
-- MAGIC Let's set up an environment where no users can create clusters.
-- MAGIC 1. In the admin console, click the **Groups** tab.
-- MAGIC 1. Select the *users* group.
-- MAGIC 1. Click the **Entitlements** tab.
-- MAGIC 1. Clear the **Allow unrestricted cluster creation** checkbox.
-- MAGIC
-- MAGIC Since entitlements are additive, we must also ensure that this entitlement isn't enabled for any other groups that our targeted users might belong to, or users themselves. That can be easily seen in the **Users** tab.
-- MAGIC
-- MAGIC Since users cannot create their own clusters under this model, it's up to a workspace administrator to create them on demand or in anticipation of user needs. Which then gives rise to a new decision: do we create one cluster for each user, or do we share them? We explore both options next.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating clusters for use by a single user
-- MAGIC
-- MAGIC Let's work through creating an example cluster for exclusive use by one of our users.
-- MAGIC
-- MAGIC 1. In the Data Science and Engineering Workspace, let's click the **Compute** icon in the left sidebar.
-- MAGIC 1. Click **Create Cluster**.
-- MAGIC 1. Let's leave **Multi node** selected. Choose **Single node** only if there is a specific need; for example:
-- MAGIC     * Some tools require single node configurations
-- MAGIC     * There exists a strong desire for the most economical configuration
-- MAGIC 1. Let's choose *Single user* for the **Access mode**.
-- MAGIC 1. For this access mode, we need to designate who will be able to connect to the cluster. Let's choose the user for whom we are creating this cluster.
-- MAGIC 1. Choose a Databricks runtime. In general, choose the most recent LTS version available, unless there is a need for a specific version. For Unity Catalog support we recommend 11.1 or greater.
-- MAGIC 1. Choose autopilot options and cluster configurations in accordance with your needs. For this example let's choose an economical setup. For more information on parameter selection, refer to <a href="https://docs.databricks.com/clusters/cluster-config-best-practices.html" target="_blank">this document</a>.
-- MAGIC 1. Finally, let's click **Create Cluster**
-- MAGIC
-- MAGIC While the cluster is being created, let's configure the permissions.
-- MAGIC 1. Click **Permissions**.
-- MAGIC 1. Delete any additional grants.
-- MAGIC 1. Select the user for whom we are creating the cluster.
-- MAGIC 1. Let's select a permission of **Can Restart**. This provides the ability to connect and restart the cluster if necessary. 
-- MAGIC 1. Click **Save**.
-- MAGIC
-- MAGIC Note that for a *Single user* cluster, only the user designated by the **Single user access** specification will be able to connect to the cluster, irrespective of the permissions.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating clusters for use by multiple users
-- MAGIC
-- MAGIC Creating shared clusters lends itself to a simpler, more economical environment. At the present time, however, shared clusters have some restrictions that might not make this a viable option for some use cases; check the <a href="https://docs.databricks.com/data-governance/unity-catalog/compute.html#what-is-cluster-access-mode" target="_blank">docs</a> for a matrix of supported features for each mode. If you need a feature only supported in the *Single user* configuration, then you will have to consider a hybrid model for allocating your clusters.
-- MAGIC
-- MAGIC Let's go ahead and create an example cluster that can be shared by everyone.
-- MAGIC 1. In the Data Science and Engineering Workspace, let's click the **Compute** icon in the left sidebar.
-- MAGIC 1. Click **Create Cluster**.
-- MAGIC 1. Let's leave **Multi node** selected.
-- MAGIC 1. Let's choose *Shared* for the **Access mode**.
-- MAGIC 1. Choose the most recent version available, unless there is a need for a specific version.
-- MAGIC 1. Choose autopilot options and cluster configurations in accordance with your needs. For this example let's choose an economical setup.
-- MAGIC 1. Finally, let's click **Create Cluster**.
-- MAGIC
-- MAGIC While the cluster is being created, let's configure the permissions.
-- MAGIC 1. Click **Permissions**.
-- MAGIC 1. Delete any additional grants.
-- MAGIC 1. Let's select the *all users* group.
-- MAGIC 1. Let's select a permission of **Can Restart**. This provides the ability to connect and restart the cluster if necessary.
-- MAGIC 1. Click **Save**.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Restricted cluster create
-- MAGIC
-- MAGIC This final scenario represents a compromise between the two extreme approaches mentioned already. In this scenario, **cluster policies** are put in place that allow users to create their own clusters, subject to restrictions set out in the policy.
-- MAGIC
-- MAGIC This approach represents a reasonable compromise between the other two approaches. The main benefit is that it eliminates potential bottlenecks by allowing users to create clusters when needed. However, policies can specify limits to help keep costs bounded. Additional benefits include:
-- MAGIC * You can fix parameters, allowing consistent clusters within allowed limits
-- MAGIC * You can impose tags for accurate cost tracking
-- MAGIC * You can hide user interface elements for fixed parameters, thereby simplifying the experience for your users
-- MAGIC
-- MAGIC The main downside to this approach is that some upfront effort is required to set this up, with a little bit of ongoing effort to maintain and update policies. And, while it does provide considerable control over cost and consistency, this approach does not generally provide as much control as preventing users from creating clusters altogether.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating a simple cluster policy
-- MAGIC
-- MAGIC
-- MAGIC Let's create a simple example policy for all users that imposes the following constraints:
-- MAGIC * Single node, single user only (and hide the associated controls)
-- MAGIC * Fix (and display) the DBR version to 11.1
-- MAGIC * Provide ability to select node type, but limited to some conservative values
-- MAGIC * Fix auto-termination to 120 minutes and hide control
-- MAGIC
-- MAGIC An example policy is provided here as a starting point. For more examples and complete documentation of available options, please refer to <a href="https://docs.databricks.com/administration-guide/clusters/index.html" target="_blank">the documentation</a>.
-- MAGIC
-- MAGIC 1. In the Data Science and Engineering Workspace, let's click the **Compute** icon in the left sidebar.
-- MAGIC 1. Click the **Cluster policies** tab.
-- MAGIC 1. Click **Create Cluster Policy**.
-- MAGIC 1. Let's specify a name for the policy (for example, *single_node*).
-- MAGIC 1. Now let's paste the JSON below into the cluster policy definition.
-- MAGIC     ```
-- MAGIC     {
-- MAGIC       "spark_conf.spark.databricks.cluster.profile": {
-- MAGIC         "type": "fixed",
-- MAGIC         "value": "singleNode",
-- MAGIC         "hidden": true
-- MAGIC       },
-- MAGIC       "spark_version": {
-- MAGIC         "type": "fixed",
-- MAGIC         "value": "11.1.x-scala2.12"
-- MAGIC       },
-- MAGIC       "node_type_id": {
-- MAGIC         "type": "allowlist",
-- MAGIC         "values": [
-- MAGIC           "i3.xlarge",
-- MAGIC           "i3.2xlarge"
-- MAGIC         ],
-- MAGIC         "defaultValue": "i3.2xlarge"
-- MAGIC       },
-- MAGIC       "autotermination_minutes": {
-- MAGIC         "type": "fixed",
-- MAGIC         "value": 120,
-- MAGIC         "hidden": true
-- MAGIC       },
-- MAGIC       "num_workers": {
-- MAGIC         "type": "fixed",
-- MAGIC         "value": 0,
-- MAGIC         "hidden": true
-- MAGIC       },
-- MAGIC       "data_security_mode": {
-- MAGIC         "type": "fixed",
-- MAGIC         "value": "SINGLE_USER",
-- MAGIC         "hidden": true
-- MAGIC       }
-- MAGIC     }
-- MAGIC     ```
-- MAGIC 1. Now let's grant access to the policy; let's click the **Permissions** tab.
-- MAGIC 1. In the **NAME** dropdown, select the *all users* group with the permission *Can Use*.
-- MAGIC 1. Click **Add**.
-- MAGIC 1. Finally, click **Create**.
-- MAGIC
-- MAGIC All users will now be able to create their own clusters. But unless they have the **Allow unrestricted cluster creation** entitlement, then their clusters will be subject to the constraints set out in this policy.
