"""
Start Spark Worker on Node 192.168.4.101
Run: python cluster/native/start_worker_101.py
"""
import subprocess
import os
import pyspark

jars = os.path.join(os.path.dirname(pyspark.__file__), "jars", "*")

subprocess.run([
    "java", "-cp", jars,
    "org.apache.spark.deploy.worker.Worker",
    "--host", "192.168.4.101",
    "--memory", "24g",
    "--cores", "20",
    "spark://192.168.4.100:7077"
])
