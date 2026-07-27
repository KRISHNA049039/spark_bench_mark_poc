"""
Download and extract Apache Spark for native cluster mode.
Run this on ALL machines.
"""

import os
import sys
import urllib.request
import tarfile
import shutil

SPARK_VERSION = "3.5.1"
HADOOP_VERSION = "3"
SPARK_URL = f"https://archive.apache.org/dist/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-hadoop{HADOOP_VERSION}.tgz"
INSTALL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark")
SPARK_HOME = os.path.join(INSTALL_DIR, f"spark-{SPARK_VERSION}-bin-hadoop{HADOOP_VERSION}")


def main():
    if os.path.exists(SPARK_HOME):
        print(f"Spark already installed at: {SPARK_HOME}")
        print("Delete the folder and re-run to reinstall.")
        return

    os.makedirs(INSTALL_DIR, exist_ok=True)
    tgz_path = os.path.join(INSTALL_DIR, f"spark-{SPARK_VERSION}.tgz")

    print(f"Downloading Spark {SPARK_VERSION}...")
    print(f"  URL: {SPARK_URL}")
    urllib.request.urlretrieve(SPARK_URL, tgz_path)
    print(f"  Downloaded: {tgz_path}")

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(INSTALL_DIR)
    os.remove(tgz_path)

    print(f"\nSpark installed at: {SPARK_HOME}")
    print(f"\nSet environment variable:")
    print(f"  set SPARK_HOME={SPARK_HOME}")
    print(f"  set PATH=%SPARK_HOME%\\bin;%PATH%")


if __name__ == "__main__":
    main()
