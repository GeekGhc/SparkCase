from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression

# 初始化SparkSession
spark = SparkSession.builder.master("local").appName("California Housing").config("spark.executor.memory",
                                                                                  "1gb").getOrCreate()

# 读取数据并创建RDD
dataRdd = spark.read.text('./california_housing/cal_housing.data').rdd
# 读取数据的每个属性的定义并创建RDD
headerRdd = spark.read.text('./california_housing/cal_housing.domain').rdd
# 字符分隔数组
rdd = dataRdd.map(lambda x: x[0].split(','))
# 转成结构化DataFrame
df = rdd.map(lambda line: Row(longitude=line[0],
                              latitude=line[1],
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5],
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()


# 字段类型转换
def convertColumn(df, columns, newType):
    for name in columns:
        df = df.withColumn(name, df[name].cast(newType))
    return df


# 定义类型转换字段
columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome', 'population',
           'totalBedRooms', 'totalRooms']
# 字段类型转换
df = convertColumn(df, columns, FloatType())

# 统计所有建造年限各有多少房子
# df.groupBy("housingMedianAge").count().sort("housingMedianAge", ascending=False).show()

# 房价字段处理
# df = df.withColumn("medianHouseValue", col("medianHouseValue") / 100000)

# 添加 每个家庭的平均房间数、每个家庭的平均人数、卧室在总房间的占比
df = df.withColumn("roomsPerHousehold", col("totalRooms") / col("households")).withColumn("populationPerHousehold",
                                                                                          col("population") / col(
                                                                                              "households")).withColumn(
    "bedroomsPerRoom", col("totalBedRooms") / col("totalRooms"))

# 指定列
df = df.select("medianHouseValue",
               "totalBedRooms",
               "population",
               "households",
               "medianIncome",
               "roomsPerHousehold",
               "populationPerHousehold",
               "bedroomsPerRoom")

# 对象分成两部分：房价和一个包含其余属性的列表
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
df = spark.createDataFrame(input_data, ["label", "features"])

# 数据标准化
standScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
scaler = standScaler.fit(df)
scaled_df = scaler.transform(df)

# print(scaled_df.take(1))

# 分割训练集和测试集
train_data, test_data = scaled_df.randomSplit([.8, .2], seed=123)

# 构建现行回归模型
lr = LinearRegression(featuresCol='features_scaled', labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lineModel = lr.fit(train_data)

# 模型评估
predicted = lineModel.transform(test_data)
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])
# 预测值和真实值机进行比较
predictionAndLabel = predictions.zip(labels).collect()
print(predictionAndLabel[:2])
