# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Infrastructure Setup Demo
# MAGIC
# MAGIC In this demo, we will provide a walkthrough of making the important architectural decisions, setting up the infrastructure for a machine learning project, and setting up an MLOps Stacks project.
# MAGIC
# MAGIC **Note**: Every organization delegates these responsibilities differently. It might be the responsibility of a platform architect, machine learning architect, machine learning engineer, or data scientist to perform these tasks. Check with your organization's leadership to determine the best way to get this infrastructure set up for a project.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC * Make important ML infrastructure and architecture decisions
# MAGIC * Create Databricks environments
# MAGIC * Set up an MLOps Stacks project
# MAGIC * Bring an MLOps Stacks project into an IDE like Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * Access to the Accounts Console to:
# MAGIC   * Create a Unity Catalog metastore
# MAGIC   * Create a Databricks Workspace
# MAGIC * Access to the Databricks CLI in a command line environment
# MAGIC * Access to a Git provider from within your Databricks workspace
# MAGIC
# MAGIC **Note**: If you do not have this access, we recommend that you follow along with the demonstration. If you have access to Databricks Academy self-paced learning courses, there is a video walkthrough within the course content.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Make Infrastructure Decisions
# MAGIC
# MAGIC We've established a few things so far:
# MAGIC
# MAGIC 1. We should be doing [data-centric machine learning](https://www.databricks.com/blog/2021/06/23/need-for-data-centric-ml-platforms.html).
# MAGIC 2. We want to have an environement for `dev`, `staging`, and `prod`.
# MAGIC 3. We want to follow a [deploy-code approach](https://docs.databricks.com/en/machine-learning/mlops/deployment-patterns.html#deploy-code-recommended).
# MAGIC
# MAGIC Next, we need to make a few decisions:
# MAGIC
# MAGIC 1. Will the project use existing infrastructure or new infrastructure?
# MAGIC 2. How many workspaces will the project use?
# MAGIC 3. What CI/CD tooling will be used?
# MAGIC
# MAGIC Let's walk through each of these decisions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision 1: New or Existing Infrastructure
# MAGIC
# MAGIC Our first decision to whether to use new or existing infrastructure for our project.
# MAGIC
# MAGIC In general, it's simpler and less work to use existing infrastructure. So while it's good to use existing infrastructure, the following *must be true*:
# MAGIC
# MAGIC * The infrastructure has a `dev`, `staging`, and `prod` environment.
# MAGIC * The infrastructure is Unity Catalog-enabled.
# MAGIC * The infrastructure is able to connect with a Git provider.
# MAGIC
# MAGIC If all three of these are true, it's a good idea to try to use existing infrastructure to save time. If not, we recommenend creating new infrastructure to meet this criteria.
# MAGIC
# MAGIC **Conclusion for Demo Only**: We'll move forward with creating new infrastructure.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision 2: Number of Workspaces
# MAGIC
# MAGIC Our next decision to determine the number of Databricks workspaces we'll use for our project.
# MAGIC
# MAGIC At a high level, we have two options here:
# MAGIC
# MAGIC 1. Create *one workspace* for `dev`, `staging`, and `prod`
# MAGIC     * Simpler infrastructure setup
# MAGIC     * More overhead in indirectly separating environments in one workspace
# MAGIC     * Could be a better option for single projects
# MAGIC 2. Create *a separate workspace* for `dev`, `staging`, and `prod`
# MAGIC     * More infrastructure setup
# MAGIC     * Less overhead in directly separating environments that are already separate
# MAGIC     * Could be a better option for multiple projects
# MAGIC
# MAGIC **Conclusion for Demo Only**: For the sake of infrastructure simplicity with respect to training environments, we'll move forward with *Option 1 - One Workspace*.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision 3: CI/CD Tooling
# MAGIC
# MAGIC Our final decision to determine what CI/CD tooling we will be using.
# MAGIC
# MAGIC As of now, we have two options for CI/CD tooling when working with MLOps Stacks:
# MAGIC
# MAGIC 1. GitHub (and GitHub Actions)
# MAGIC 2. Azure DevOps
# MAGIC
# MAGIC **Conclusion for Demo Only**: To be general and broadly applicable, we'll move forward with GitHub and GitHub Actions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Databricks Environments
# MAGIC
# MAGIC Now that we've made key decisions, we need to move forward with setting up our infrastructure (if we're creating new infrastructure!).
# MAGIC
# MAGIC These are the things that we neet to set up:
# MAGIC
# MAGIC 1. A Unity Catalog metastore
# MAGIC 2. 1 or 3 Databricks workspaces
# MAGIC 3. `dev`, `test`, `staging`, and `prod` Unity Catalog environments – the `test` catalog is used by MLOps Stacks
# MAGIC 4. Create a service principal and give it  permissions to use each catalog
# MAGIC
# MAGIC **Note**: While we'll be doing this manually in this demo, a lot of it can be [completed using Terraform](https://docs.databricks.com/en/dev-tools/index-iac.html) for a more scalable approach.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting Up a Unity Catalog metastore
# MAGIC
# MAGIC Our first step is setting up a Unity Catalog metastore.
# MAGIC
# MAGIC As a reminder, a [Unity Catalog metastore](https://docs.databricks.com/en/data-governance/unity-catalog/index.html#the-unity-catalog-object-model) is the top-level container for your data. Each metastore exposes a three-level namespace (catalog.schema.table) that organizes your data, and we'll create the appropriate catalogs later.
# MAGIC
# MAGIC You can follow the below steps to create a Unity Catalog metastore:
# MAGIC
# MAGIC 1. Navigate to the **[Accounts Console](https://accounts.cloud.databricks.com)** (note: you *will* need access).
# MAGIC 2. Click the **Data** tab on the left-side navigation bar.
# MAGIC 3. Click the **Create metastore** button on the right side of the page.
# MAGIC 4. Name your metastore "mlops".
# MAGIC 5. Select your region. *Note:* this the region where your workspaces must be located.
# MAGIC 6. Input an object storage location and an associated cloud role to access the location. You might need help from our cloud platform administrator for this.
# MAGIC 6. Click **Create**.
# MAGIC 7. For now, we will skip adding metastores to any existing workspaces.
# MAGIC
# MAGIC **Note**: One of the key features of Unity Catalog metastores is that they can be assigned to multiple workspaces. This means that if you're using the three-workspace setup, you can access your data across all three workspaces.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting Up a Databricks Workspace
# MAGIC
# MAGIC Our next step is setting up a Databricks workspace. We're only going to set up one, but you can follow these instructions to set up separate `dev`, `staging`, and `prod` workspaces.
# MAGIC
# MAGIC You can follow the below steps to create a Databricks workspace:
# MAGIC
# MAGIC 1. Navigate to the **[Accounts Console](https://accounts.cloud.databricks.com)**.
# MAGIC 2. Click the **Workspaces** tab on the left-side navigation bar.
# MAGIC 3. Click the **Create workspace** button on the right side of the page.
# MAGIC 4. Enter your workspace name. We will use `mlo` for "machine learning operations".
# MAGIC 5. Select your region, credential configuration, and storage configuration.
# MAGIC 6. Enable **Unity Catalog**.
# MAGIC 7. Select the **Metastore** that was previously created (in our case, "mlops").
# MAGIC 8. Click **Save**.
# MAGIC
# MAGIC **Note**: For later use of the Databricks CLI, you will need to [authenticate](https://docs.databricks.com/en/dev-tools/auth/pat.html) your CLI to work with your new Databricks Workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up `dev`, `test`, `staging`, and `prod` Unity Catalog catalogs
# MAGIC
# MAGIC Whether we're using a single-workspace setup or three-workspace setup, we need to create our four new Unity Catalog catalogs for our metastore.
# MAGIC
# MAGIC For `dev`, `test`, `staging`, and `prod`, follow the below steps:
# MAGIC
# MAGIC 1. Navigate to the workspace you created.
# MAGIC 2. Click on the **Catalog** tab on the left-side navigation bar.
# MAGIC 3. Click **Create Catalog**, name it (`dev`, `test`, `staging`, and `prod`) and use the default metastore root storage location.
# MAGIC
# MAGIC Next, within each of those catalogs, we want to create project schemas. This project schema will be called `mlops_demo` in our case.
# MAGIC
# MAGIC When we are done, we should have four catalogs: `dev`, `test`, `staging`, and `prod`. Each of these catalogs should have a schema called `mlops_demo`.
# MAGIC
# MAGIC This means our eventual data will be in the following locations:
# MAGIC
# MAGIC * `mlops` metastore: `prod.mlops_demo.<table_name>`
# MAGIC * `mlops` metastore: `staging.mlops_demo.<table_name>`
# MAGIC * `mlops` metastore: `test.mlops_demo.<table_name>`
# MAGIC * `mlops` metastore: `dev.mlops_demo.<table_name>`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create and Configure a Service Principal
# MAGIC
# MAGIC Our final step in Databricks to to create and configure our a service principal for our CI/CD tooling.
# MAGIC
# MAGIC Our CI/CD tooling will use this service principal to authenticate with Databricks, and Databricks will use it to govern the CI/CD tooling's access with Databricks environments.
# MAGIC
# MAGIC 1. To create the service principal, we can follow [these instructions](https://docs.databricks.com/en/administration-guide/users-groups/service-principals.html#add-service-principals-to-your-account-using-the-account-console). We'll name ours `mlo_sp`.
# MAGIC 2. Next, we will need to [assign the service principal to our workspace(s)](https://docs.databricks.com/en/administration-guide/users-groups/service-principals.html#assign-a-service-principal-to-a-workspace-using-the-account-console).
# MAGIC 3. Finally, we will want to [grant the service principal access to our catalogs](https://docs.databricks.com/en/catalog-explorer/manage-permissions.html#grant-permissions-on-objects-in-a-unity-catalog-metastore).
# MAGIC
# MAGIC **Note**: You will need to be an Account Admin to create service principal following the above instructions. If you are not an Account Admin, ask your Account Admin to set up a service principal for you.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Set Up an MLOps Stacks Project
# MAGIC
# MAGIC Our next step is to set up an [MLOps Stacks](https://docs.databricks.com/en/machine-learning/mlops/mlops-stacks.html) project.
# MAGIC
# MAGIC Before moving forward with this, please ensure:
# MAGIC
# MAGIC 1. You have access to a command line environment.
# MAGIC 2. The Databricks CLI has been [installed](https://docs.databricks.com/en/dev-tools/cli/install.html) within your command line environment.
# MAGIC 3. You have [authenticated](https://docs.databricks.com/en/dev-tools/cli/authentication.html) the Databricks CLI to work with your newly created workspace.
# MAGIC
# MAGIC Once you've set up the above requirements, you will be able to move forward with creating an MLOps Stacks project.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initiate the Project Setup
# MAGIC
# MAGIC To get started, you should first navigate to a folder location in your command line environment where you want to project to live. 
# MAGIC
# MAGIC Next, we will run the Databricks CLI code below to initate the MLOps Stacks project set up in our terminal environment:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```
# MAGIC databricks bundle init mlops-stacks
# MAGIC ```
# MAGIC
# MAGIC This will launch a series of on-screen prompts.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Answer the On-screen Prompts
# MAGIC
# MAGIC Answering these on-screen prompts is really important. This will help set up your project correctly, so you have less manual work to do later.
# MAGIC
# MAGIC The responses to the prompts for this demo are below:
# MAGIC
# MAGIC * **CI/CD or Project or Both**: Both
# MAGIC * **Project Name**: `mlops_demo`
# MAGIC * **Root directory name**: `mlops_demo`
# MAGIC * **Select cloud**: Select cloud you are using: `azure` or `aws`. Note that Google Cloud is not supported at this time. We'll use `aws`.
# MAGIC * **Select CICD platform**: `github_actions`
# MAGIC * **Staging workspace URL**: Your staging workspace URL, for example `https://my-databricks-environment.cloud.databricks.com`. Our case in `curriculum-mlo`.
# MAGIC * **Production workspace URL**: Your staging workspace URL, for example `https://my-databricks-environment.cloud.databricks.com`. Our case is `curriclum-mlo`.
# MAGIC * **Name of default branch**: Hit <kbd>return</kbd> to accept the default of `main`
# MAGIC * **Name of the release branch**: Hit <kbd>return</kbd> to accept the default of `release`
# MAGIC * **Read group**: Hit <kbd>return</kbd> to accept the default of `users`
# MAGIC * **Model Registry with Unity Catalog**: Select `yes`
# MAGIC * **Schema name**: The name of the project, `mlops_demo`
# MAGIC * **Unity Catalog group**: Hit <kbd>return</kbd> to accept the default of `account users`
# MAGIC * **Include Feature Store**: Select `yes`
# MAGIC
# MAGIC At this point, your project will be created in its own directory.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Bring Project into IDE
# MAGIC
# MAGIC Once your MLOps Stacks project is set up locally, you'll want to be able to access it within your integrated development environment or IDE.
# MAGIC
# MAGIC If your IDE is local, you should be able to access it directly from your IDE. So you can skip the last step in this section ("Clone the Project to Databricks").
# MAGIC
# MAGIC If your IDE is in the cloud (like Databricks), you'll want to follow all of the below steps to bring your project into your cloud-based IDE.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Git Repository
# MAGIC
# MAGIC First, you will initialize your MLOps Stacks project as a local Git repository.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC git init --initial-branch=main
# MAGIC ```
# MAGIC
# MAGIC Next, we'll commit our basic resources that don't contain ML code:
# MAGIC
# MAGIC ```sh
# MAGIC git add README.md docs .gitignore mlops_demo/resources/README.md
# MAGIC git commit -m "Adding project README"
# MAGIC ```
# MAGIC
# MAGIC Next, we will create and check out a new `dev` branch.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC git checkout -b dev
# MAGIC ```
# MAGIC
# MAGIC And finally, we'll commit all of code assets created by MLOps Stacks into the `dev` branch.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC git add .
# MAGIC git commit -m "Adding all assets"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Centralized Repo in a Git Provider
# MAGIC
# MAGIC #### Create the Repo
# MAGIC
# MAGIC First, you will need to create a repository in your Git provider that matches the name of your repository.
# MAGIC
# MAGIC In this demo, we'll create a repository named `mlops_demo`.
# MAGIC
# MAGIC #### Set the Remotes
# MAGIC
# MAGIC Next, we will need set the remotes of the local repository to the address of the repository.
# MAGIC
# MAGIC In this demo, we are using:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC git remote add upstream git@github.com:databricks-academy/mlops_demo
# MAGIC ```
# MAGIC
# MAGIC **Note**: Note that your method of setting the origin here might differ depending on your Git provider, your organization's Git provider setup, and your own authentication with your Git provider.
# MAGIC
# MAGIC #### Push Code to the Git Provider
# MAGIC
# MAGIC And next, we'll push all of our code the Git provider repository.
# MAGIC
# MAGIC ```sh
# MAGIC git push --all upstream
# MAGIC ```
# MAGIC
# MAGIC **Note**: This method will push all branches to the Git provider repository.
# MAGIC
# MAGIC #### Reviewing the Repo
# MAGIC
# MAGIC It's useful to take the time to take a look at the repository in the Git provider.
# MAGIC
# MAGIC Look at things like:
# MAGIC
# MAGIC * README
# MAGIC * Project structure
# MAGIC * Tips on using the repo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up Service Principal Token
# MAGIC
# MAGIC We have one more setup step to do related to our service principal and our Git provider: we need to configure the service principal as an access key for the Git provider when working with our Databricks Workspace(s). Luckily, MLOps Stacks makes this pretty easy.
# MAGIC
# MAGIC We do this by:
# MAGIC
# MAGIC 1. Creating a Service Principal Token
# MAGIC 2. Adding the token as a Git provider secret for each workspace to work with MLOps Stacks
# MAGIC
# MAGIC #### Create the Service Principal Token
# MAGIC
# MAGIC To create the service principal token, we'll want to run the following in our command line environment:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ```sh
# MAGIC databricks token-management create-obo-token <application-id> <lifetime-seconds>
# MAGIC ```
# MAGIC
# MAGIC **Note**: We can get our `application-id` from the Service Principal page in our Workspace.
# MAGIC
# MAGIC This will return a JSON structure. Be sure to copy the `"token_value"` field's value – you'll use this in the next step.
# MAGIC
# MAGIC #### Add the Token to Git Provider
# MAGIC
# MAGIC Once the token is generated and copied, we'll add the token value to our Git provider.
# MAGIC
# MAGIC For this demo, we'll do the following in GitHub:
# MAGIC
# MAGIC 1. Go to our GitHub repository's settings page
# MAGIC 2. Click on the "Secrets and variables" tab on the left-side navigation bar and the "Actions" dropdown option
# MAGIC 3. Add a new repository secret for the staging and production workspaces:
# MAGIC     * `PROD_WORKSPACE_TOKEN`
# MAGIC     * `STAGING_WORKSPACE_TOKEN`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clone the Project to Databricks
# MAGIC
# MAGIC The final step here is to bring the repo into Databricks or another IDE.
# MAGIC
# MAGIC **Note**: You can skip this step if your IDE is local to your command line environment.
# MAGIC
# MAGIC For Databricks, we can do this by completing the following steps:
# MAGIC
# MAGIC 1. Click the **Workspace** tab in the left-side navigation bar
# MAGIC 2. Click the **Repos** section of the Workspaces page
# MAGIC 3. Click your user folder and then the **Add Repo** button in the Repos page 
# MAGIC 4. Enter your repository's URL and click **Create Repo**
# MAGIC
# MAGIC **Note**: You might need to configure authentication between your Git provider and Databricks. This will depend on your organization's Git provider setup.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Summary
# MAGIC
# MAGIC In this demonstration, you learned how to:
# MAGIC
# MAGIC * Make important ML infrastructure and architecture decisions
# MAGIC * Create Databricks environments
# MAGIC * Set up an MLOps Stacks project
# MAGIC * Bring an MLOps Stacks project into an IDE like Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>