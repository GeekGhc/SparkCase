from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, DataType

# 初始化SparkSession
spark = SparkSession.builder.master("local").appName("California Housing").config("spark.executor.memory",
                                                                                  "1gb").getOrCreate()

# 读取数据并创建RDD
rdd = spark.read.text('./california_housing/cal_housing.data').rdd
# 读取数据的每个属性的定义并创建RDD
headerRdd = spark.read.text('./california_housing/cal_housing.domain').rdd
# 字符分隔数组
rdd = rdd.map(lambda x: x[0].split(','))
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
def convertColumn(df, names, newType):
    for name in names:
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
df = df.withColumn("medianHouseValue", col("medianHouseValue") / 100000)
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
df.show()
