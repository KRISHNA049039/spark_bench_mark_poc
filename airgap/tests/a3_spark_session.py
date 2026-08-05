import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("airgap-smoke") \
    .config("spark.driver.memory", "1g") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

total = spark.sparkContext.parallelize(range(100)).sum()
spark.stop()
assert total == 4950, f"wrong sum: {total}"
print(f"      sum(0..99)={total}  OK")
