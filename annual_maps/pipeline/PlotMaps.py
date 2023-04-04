# Databricks notebook source


# COMMAND ----------



# COMMAND ----------

my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': ml_outputs_df_xy, 'attributes': [pollutant]})
display(my_map)

# COMMAND ----------



# COMMAND ----------


