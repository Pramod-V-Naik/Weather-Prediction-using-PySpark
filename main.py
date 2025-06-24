import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType
from datetime import datetime, timedelta

# Step 1: Initialize Spark session
spark = SparkSession.builder.master("local[*]").appName("WeatherPrediction").getOrCreate()

# Load dataset from a file
data_path = "BDA.csv"  # Update with your file path
weather_df = spark.read.csv(data_path, header=True, inferSchema=True)

# Step 2: Feature Engineering - StringIndexer for weather conditions
indexer = StringIndexer(inputCol="weather_condition", outputCol="weather_condition_index")
weather_df = indexer.fit(weather_df).transform(weather_df)

# Step 3: Assemble features
assembler = VectorAssembler(
    inputCols=["temperature(c)", "humidity(%)", "rainfall(mm)"],
    outputCol="features"
)

# Step 4: Train Random Forest model for weather conditions prediction
rf_weather = RandomForestClassifier(labelCol="weather_condition_index", featuresCol="features")
weather_df_weather = assembler.transform(weather_df)
weather_model = rf_weather.fit(weather_df_weather)

# Step 5: Get the most recent data for trends
latest_data = weather_df.orderBy("date", ascending=False).limit(1).collect()[0]
latest_temp, latest_humidity, latest_rainfall = latest_data["temperature(c)"], latest_data["humidity(%)"], latest_data["rainfall(mm)"]
last_date = datetime.strptime(latest_data["date"], "%d-%m-%Y")

# Step 6: Predict the next 5 days of weather conditions
def predict_weather_condition(rainfall):
    """Custom logic for weather condition prediction based on rainfall thresholds."""
    if rainfall > 10:
        return "Rainy"
    elif 5 < rainfall <= 10:
        return "Cloudy"
    elif rainfall <= 5:
        return "Clear"

# Register the custom logic as a UDF
predict_weather_condition_udf = udf(predict_weather_condition, StringType())

# Generate the next 5 days of predictions
next_5_days = []
for i in range(1, 6):
    next_date = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
    temp = latest_temp + (0.1 * i)
    humidity = latest_humidity + (0.2 * i)
    rainfall = latest_rainfall + (2 * i)
    next_5_days.append((next_date, temp, humidity, rainfall))

# Create a DataFrame for the next 5 days
next_5_days_df = spark.createDataFrame(next_5_days, ["Date", "Temperature (C)", "Humidity (%)", "Rainfall (mm)"])

# Apply the UDF to predict weather condition
next_5_days_df = next_5_days_df.withColumn(
    "Predicted Weather Condition",
    predict_weather_condition_udf(col("Rainfall (mm)"))
)

# Display the predictions for the next 5 days
print("Weather prediction for the next 5 days:")
next_5_days_df.show(truncate=False)

# Step 7: Predict weather for a user-specified future date
input_date = input("Enter a future date (DD-MM-YYYY) to predict the weather: ")
input_date_obj = datetime.strptime(input_date, "%d-%m-%Y")
days_diff = (input_date_obj - last_date).days

# Calculate features based on trends
predicted_temp = latest_temp + (0.1 * days_diff)
predicted_humidity = latest_humidity + (0.2 * days_diff)
predicted_rainfall = latest_rainfall + (0.5 * days_diff)

# Create a DataFrame for the input date prediction
input_data = [(predicted_temp, predicted_humidity, predicted_rainfall)]
input_df = spark.createDataFrame(input_data, ["temperature(c)", "humidity(%)", "rainfall(mm)"])

# Predict weather condition for the input date
input_df_weather = assembler.transform(input_df)
input_weather_predictions = weather_model.transform(input_df_weather)

# Map predictions to weather conditions and adjust based on rainfall threshold
input_weather_predictions = input_weather_predictions.withColumn(
    "predicted_weather_condition_label",
    when(col("prediction") == 0.0, "Clear")
    .when(col("prediction") == 1.0, "Cloudy")
    .when(col("prediction") == 2.0, "Rainy")
    .otherwise("Unknown")
)

final_input_prediction = input_weather_predictions.withColumn(
    "final_weather_condition",
    when(col("rainfall(mm)") > 10, "Rainy")
    .when((col("rainfall(mm)") > 5) & (col("rainfall(mm)") <= 10), "Cloudy")
    .when(col("rainfall(mm)") <= 5, "Clear")
    .otherwise(col("predicted_weather_condition_label"))
)

# Display the final prediction
final_result = final_input_prediction.select(
    col("final_weather_condition").alias("Predicted Weather Condition"),
    col("temperature(c)").alias("Temperature (C)"),
    col("humidity(%)").alias("Humidity (%)"),
    col("rainfall(mm)").alias("Rainfall (mm)")
)

print("Weather prediction for :",input_date)
final_result.show(truncate=False)
