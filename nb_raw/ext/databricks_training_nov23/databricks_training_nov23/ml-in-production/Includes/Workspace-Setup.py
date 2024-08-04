# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

from dbacademy.dbhelper.workspace_helper_class import WorkspaceHelper
apiclient = WorkspaceHelper.DBAcademyRestClient()

# COMMAND ----------

WorkspaceHelper.DBAcademyRestClient.api(apiclient,
                                        'PATCH',
                                        '/2.0/workspace-conf',
                                        {
                                            "enableTokensConfig": "true"
                                        })

# COMMAND ----------

WorkspaceHelper.DBAcademyRestClient.api(apiclient,
                                        'PATCH',
                                        '/2.0/permissions/authorization/tokens',
                                        {
                                            "access_control_list": [
                                                {
                                                    "group_name": "users",
                                                    "permission_level": "CAN_USE"
                                                }
                                            ]
                                        })

# COMMAND ----------

lesson_config.create_schema = False

DA = DBAcademyHelper(course_config, lesson_config)
DA.install_datasets(reinstall_datasets=False)
