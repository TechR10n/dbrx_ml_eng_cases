-- Databricks notebook source
-- INCLUDE_HEADER_TRUE
-- INCLUDE_FOOTER_TRUE

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Managing SQL warehouses
-- MAGIC
-- MAGIC In this lab, you will learn how to:
-- MAGIC * Implement some common schemes to control compute costs associated with SQL warehouses
-- MAGIC * Pre-create SQL warehouses for users

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Prerequisites
-- MAGIC
-- MAGIC If you would like to follow along with this lab, you will need workspace admin capabilities.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Overview
-- MAGIC
-- MAGIC Like all-purpose clusters, SQL warehouses  represent a family of compute resources that your users will rely on heavily at all stages of the development lifecycle, particularly for those focused on data analytics using Databricks SQL. Without careful management, there can be significant cost associated with the compute resources to support the environment. Managing compute resources therefore represents a trade-off between convenience and cost control. Management strategies for SQL warehouses can be quantized into two broad approaches:
-- MAGIC
-- MAGIC * Unrestricted SQL warehouse creation
-- MAGIC * Disallow SQL warehouse creation
-- MAGIC
-- MAGIC There are pros and cons to each, which we will cover in this lab.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Unrestricted SQL warehouse creation
-- MAGIC
-- MAGIC In this scenario, all users are granted unrestricted capability to create their own SQL warehouses when needed.
-- MAGIC
-- MAGIC The benefits and downsides to this approach are reflective of what we see for all-purpose clusters. While it's easy to implement and maintain, it comes at a potentially high cost. Furthermore, accurate cost-tracking can be challenging.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Enabling unrestricted SQL warehouse creation
-- MAGIC
-- MAGIC Unrestricted SQL warehouse creation is accomplished in a similar manner as unrestricted cluster creation: through the **Allow unrestricted cluster creation** entitlement. Let's set up an environment where all user can create warehouses in an unrestricted manner.
-- MAGIC
-- MAGIC 1. In the admin console, click the **Groups** tab.
-- MAGIC 1. Select the *users* group.
-- MAGIC 1. Click the **Entitlements** tab.
-- MAGIC 1. Enable **Allow unrestricted cluster creation**. This entitlement applies to SQL warehouses as well.
-- MAGIC 1. Clear the **Workspace access** entitlement. If the group has the **Workspace access** entitlement, then they will be able to access the Data Science and Engineering workspace, where they will be able to create interactive clusters in an unrestricted manner. If this is not desirable, we must clear the **Workspace access** checkbox.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Complete SQL warehouse create restriction
-- MAGIC
-- MAGIC This scenario represents the complete opposite of the previous one. No users (with the exception of workspace administrators) are permitted to create SQL warehouses. Rather, administrators create them on behalf of the users, either on demand or by anticipating needs and prepopulating the environment.
-- MAGIC
-- MAGIC Since this approach is the opposite of the previous one, the pros and cons are reversed as well. The main benefit of this approach is that it delivers a predictable cost; resources can be allocated with a budget in mind, and the organization is essentially guaranteed not to exceed that budget. Administrators can also ensure consistency and tagging to help track costs.
-- MAGIC
-- MAGIC The downside to this approach is that it requires upfront and/or ongoing effort on the part of the workspace administrators to support, and potentially introduces bottlenecks and frustration if users can't get the resources they need when they need them.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Preventing user SQL warehouse creation
-- MAGIC
-- MAGIC Let's set up an environment where no users can create SQL warehouses.
-- MAGIC 1. In the admin console, click the **Groups** tab.
-- MAGIC 1. Select the *users* group.
-- MAGIC 1. Click the **Entitlements** tab.
-- MAGIC 1. Clear the **Allow unrestricted cluster creation** checkbox.
-- MAGIC 1. Ensure that this entitlement isn't enabled for any other groups. In the previous section, we enabled this entitlement for the *users* group, so let's clear that.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Creating SQL warehouses for users
-- MAGIC
-- MAGIC Since users cannot create their own SQL warehouses under this model, it's up to a workspace admin to create them on demand or in anticipation of user needs.
-- MAGIC
-- MAGIC Let's work through creating an example SQL warehouse on behalf of our users. Since all SQL warehouses are shareable, it can be shared across a group or your entire user base.
-- MAGIC
-- MAGIC 1. In Databricks SQL, let's click the **SQL Warehouses** icon in the left sidebar.
-- MAGIC 1. Click **Create SQL Warehouse**.
-- MAGIC 1. Now let's specify a name. Choose a naming scheme to make it easy for you to manage.
-- MAGIC 1. Choose size and scaling options in accordance with your needs. Refer to the <a href="https://docs.databricks.com/sql/admin/sql-endpoints.html#" target="_blank">documentation</a> for more information on the available options. For this example let's choose an economical setup.
-- MAGIC 1. In the **Advanced options**, ensure that **Unity Catalog** is enabled.
-- MAGIC 1. Finally, let's click **Create**.
-- MAGIC
-- MAGIC While the SQL warehouse is being created, let's configure the permissions.
-- MAGIC 1. Click in the text field, which also functions as a drop-down. Use it to select the desired principal; this can be a user or a group. Let's create this warehouse for use by *All users* group.
-- MAGIC 1. For permission, select **Can use**.
-- MAGIC 1. Let's click **Add**.
