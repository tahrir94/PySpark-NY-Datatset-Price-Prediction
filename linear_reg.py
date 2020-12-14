from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import six
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd

#create spark context and pass it in to SQL context, then read csv as spark dataframe using the SQL context
sc= SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('AB_NYC_2019.csv')

#drop 'bad' data - irrelevant to predicting price or lots of empty values. Then fill all null values with 0.
df_drop = df.drop("id","name","host_id","host_id","host_name","latitude","longitude","last_review")

df_fill = df_drop.na.fill(0)

#use Stringindexer to encode columns with string values to numerical representation so that the regression algorithm can process it.
indexer1 = StringIndexer(inputCol="neighbourhood_group", outputCol="ng_indexed")
df_i1 = indexer1.fit(df_fill).transform(df_fill)

indexer2 = StringIndexer(inputCol="neighbourhood", outputCol="n_indexed")
df_i2 = indexer2.fit(df_i1).transform(df_i1)

indexer3 = StringIndexer(inputCol="room_type", outputCol="room_indexed")
df_i3 = indexer3.fit(df_i2).transform(df_i2)

#print(df_i3.select('ng_indexed','n_indexed','room_indexed').show(10))

#cast all string type columns into numerical data types for correlation calculation.
df_cast = df_i3.select(df_i3.neighbourhood_group.cast("int"),
df_i3.neighbourhood.cast("int"),
df_i3.room_type.cast("int"),
df_i3.price.cast("int"),
df_i3.minimum_nights.cast("int"),
df_i3.number_of_reviews.cast("int"),
df_i3.reviews_per_month.cast("float"),
df_i3.calculated_host_listings_count.cast("int"),
df_i3.availability_365.cast("int"))

#create text file and append the correlation output of each selected column with price to the text file.
f = open("correlation.txt", "a")
f.truncate(0)

for i in df_cast.columns:
    if not( isinstance(df_cast.select(i).take(1)[0][0], six.string_types)):
        f.write("%s %s %f\n" % ("Correlation to price for ", i, df_cast.stat.corr('price',i)))

f.close()   

#cast the string data columns to numerical types again, this time including the indexed columns so that the vector assembler can process it.
df_cast2 = df_i3.select(df_i3.price.cast("int"),
df_i3.minimum_nights.cast("int"),
df_i3.number_of_reviews.cast("int"),
df_i3.reviews_per_month.cast("float"),
df_i3.calculated_host_listings_count.cast("int"),
df_i3.ng_indexed.cast("float"),
df_i3.n_indexed.cast("float"),
df_i3.room_indexed.cast("float"),
df_i3.availability_365.cast("int"))

df_final = df_cast2.na.fill(0)

#print(df_final.printSchema())
#print(df_final.dtypes)

#create vector assembler with input features and target label(price) and pass in the final transformed dataframe to it.
vectorAssembler = VectorAssembler(inputCols = ['ng_indexed', 'n_indexed', 'room_indexed', 'minimum_nights', 'number_of_reviews', 'reviews_per_month','calculated_host_listings_count','availability_365'], outputCol = 'features')
vector_df = vectorAssembler.transform(df_final)
vector_df = vector_df.select(['features', 'price'])
vector_df.show(10)

#split vectored dataframe into train data and test data
splits = vector_df.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]

#print(train.printSchema())

train_df = train.na.drop
test_df = test.na.drop

#create instance of linear regression model and feed training data to it. 
lr = LinearRegression(featuresCol = 'features', labelCol='price', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)

#compute and print values ofRMSE and R squared of training data
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r^2: %f" % trainingSummary.r2)

#show count,mean,max,min, and standard deviation of prices in training data to compare with RMSE
train_df.describe().show()

#feed test data to regression model and show prediction vs actual price for top 20 rows
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","price","features").show(20)

#use RegressionEvaluator to compute and print values of R squared and RMSE after making predictions on test data
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="price",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
