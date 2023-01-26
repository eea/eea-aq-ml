# Databricks notebook source
# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Initialize a Context Dictionary with useful data.
context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML models/algorithms
# MAGIC > **NOTE:** The examples below show simplified parametrisation of SGD, NN and XGBoost
# MAGIC > 
# MAGIC <br/>

# COMMAND ----------

# SGD

from sklearn.linear_model import SGDRegressor
model = SGDRegressor(penalty='elasticnet', alpha=0.0001, max_iter=1000) # define model parameters

# COMMAND ----------

# Relatively simple NN

tf.random.set_seed(1234)
model = Sequential(
    [
        tf.keras.layers.Dense(9, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.03)),
        tf.keras.layers.Dense(7, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.03)),
        tf.keras.layers.Dense(3, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.03)),
        tf.keras.layers.Dense(1, activation = 'relu')
    ], name= None
)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
# model.fit(
#     X_train, y_train,
#     epochs=50
# )
# model.summary()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


