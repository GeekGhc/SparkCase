from pyspark.sql import SparkSession

# 初始哈SparkSession
spark = SparkSession.builder.appName("wordCount").getOrCreate()

text_file = spark.read.text("./word.txt").rdd.map(lambda r: r[0])
counts = text_file.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

output = counts.collect()
for (word, count) in output:
    print("%s: %i " % (word, count))

spark.stop()
