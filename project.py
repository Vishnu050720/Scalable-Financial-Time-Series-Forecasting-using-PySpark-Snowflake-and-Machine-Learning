# =========================================================
# FINAL COFFEE PRICE BIG DATA PIPELINE
# Yahoo → PySpark → ML → S3 → Snowflake
# =========================================================



from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag, avg, row_number, round
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import yfinance as yf
import pandas as pd

# Create Spark session
spark = SparkSession.builder \
    .appName("CoffeePriceBigDataMLPipeline") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Download coffee price data
symbol = "KC=F"
data = yf.download(symbol, start="2014-01-01", end="2026-02-01")
data.reset_index(inplace=True)

# Remove ticker level from columns
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# Convert pandas to Spark DataFrame
df = spark.createDataFrame(data)

# Data cleaning
df_clean = df \
    .dropna() \
    .withColumn("DATE", to_date(col("Date"))) \
    .withColumn("OPEN", round(col("Open").cast(DoubleType()), 2)) \
    .withColumn("HIGH", round(col("High").cast(DoubleType()), 2)) \
    .withColumn("LOW", round(col("Low").cast(DoubleType()), 2)) \
    .withColumn("CLOSE", round(col("Close").cast(DoubleType()), 2)) \
    .withColumn("VOLUME", round(col("Volume").cast(DoubleType()), 2)) \
    .withColumn("PRICE", col("CLOSE")) \
    .select("DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "PRICE")

# Feature engineering
windowSpec = Window.orderBy("DATE")
window7 = Window.orderBy("DATE").rowsBetween(-6, 0)
window14 = Window.orderBy("DATE").rowsBetween(-13, 0)

df_features = df_clean \
    .withColumn("PREV_PRICE", lag("PRICE", 1).over(windowSpec)) \
    .withColumn("PRICE_CHANGE", col("PRICE") - col("PREV_PRICE")) \
    .withColumn("MA_7", avg("PRICE").over(window7)) \
    .withColumn("MA_14", avg("PRICE").over(window14)) \
    .dropna()

# Prepare machine learning dataset
assembler = VectorAssembler(
    inputCols=[
        "OPEN", "HIGH", "LOW", "VOLUME",
        "PREV_PRICE", "PRICE_CHANGE",
        "MA_7", "MA_14"
    ],
    outputCol="features"
)

ml_data = assembler.transform(df_features)

final_data = ml_data.select(
    "DATE",
    col("PRICE").alias("label"),
    "features"
)

# Chronological 80-20 split
window_row = Window.orderBy("DATE")
final_data = final_data.withColumn(
    "row_num",
    row_number().over(window_row)
)

total_count = final_data.count()
split_index = int(total_count * 0.8)

train_data = final_data.filter(col("row_num") <= split_index).drop("row_num")
test_data = final_data.filter(col("row_num") > split_index).drop("row_num")

# Train Random Forest model
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    numTrees=150,
    maxDepth=7,
    seed=42
)

model = rf.fit(train_data)
predictions = model.transform(test_data)

# Round predictions
predictions = predictions.withColumn(
    "prediction", round(col("prediction"), 2)
)

predictions.select("DATE", "label", "prediction").show(10)

# Model evaluation
rmse_eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

mae_eval = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

rmse = rmse_eval.evaluate(predictions)
mae = mae_eval.evaluate(predictions)

print("MODEL PERFORMANCE")
print("RMSE:", rmse)
print("MAE :", mae)

# Save predictions to S3
predictions.select("DATE", "label", "prediction") \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv("s3://aws-glue-assets-419659069697-us-east-1/test_output/")

print("Predictions saved to S3")

# Push predictions to Snowflake
sfOptions = {
    "sfURL": "PLTEBCV-TE35146.snowflakecomputing.com",
    "sfUser": "VISHNU0507",
    "sfPassword": "Vishnu@1234567",
    "sfDatabase": "COFFEE_DB",
    "sfSchema": "PUBLIC",
    "sfWarehouse": "COMPUTE_WH",
    "sfRole": "ACCOUNTADMIN"
}


predictions.select("DATE", "label", "prediction") \
    .write \
    .format("net.snowflake.spark.snowflake") \
    .options(**sfOptions) \
    .option("dbtable", "COFFEE_PRICE_PREDICTIONS") \
    .mode("overwrite") \
    .save()

print("Predictions successfully pushed to Snowflake")
