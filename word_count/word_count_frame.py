from pyspark.sql import SparkSession
from pyspark.sql.functions import *

if __name__ == "__main__":
    spark = SparkSession.builder.appName("wordCount").getOrCreate()

    lines = spark.read.text("./word.txt")
    wordCounts = lines.select(explode(split(lines.value, " "))).groupBy("col").count()

    wordCounts.show()
    spark.stop()
