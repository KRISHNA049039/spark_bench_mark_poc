"""
Hybrid Cluster Runner — Native Driver + Docker Workers

Run this on the MASTER machine (host, not Docker).
Workers should already be running in Docker on other machines.

This script:
1. Starts a Spark master on this host
2. Waits for workers to connect
3. Runs the 3-phase benchmark
4. Generates the comparison report

Usage:
    python cluster/hybrid/run_cluster.py

Requires: pip install pyspark torch torchvision psutil numpy pandas scikit-learn
"""

import os
import sys
import time
import subprocess
import threading
import signal

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Configuration — CHANGE THESE
MASTER_IP = "192.168.4.100"  # This machine's LAN IP
MASTER_PORT = 7077
DRIVER_PORT = 33000

# Set environment for the benchmark
os.environ["SPARK_MASTER"] = f"spark://{MASTER_IP}:{MASTER_PORT}"
os.environ["SPARK_LOCAL_HOSTNAME"] = MASTER_IP
os.environ["SPARK_DRIVER_HOST"] = MASTER_IP
os.environ["SPARK_DRIVER_PORT"] = str(DRIVER_PORT)
os.environ["SPARK_DRIVER_BLOCKMANAGER_PORT"] = "33005"
os.environ["SPARK_PUBLIC_DNS"] = MASTER_IP
os.environ["SPARK_DRIVER_MEMORY"] = "4g"
os.environ["SPARK_EXECUTOR_MEMORY"] = "12g"
os.environ["SPARK_EXECUTOR_MEMORY_OVERHEAD"] = "2g"
os.environ["SPARK_EXECUTOR_CORES"] = "4"
os.environ["SPARK_NUM_EXECUTORS"] = "4"
os.environ["BENCHMARK_OUTPUT_DIR"] = os.path.join(PROJECT_ROOT, "benchmark_results")
os.environ["BENCHMARK_SAMPLES"] = "1000"
os.environ["BENCHMARK_BATCH_SIZE"] = "64"
os.environ["BENCHMARK_PARTITIONS"] = "8"
os.environ["FORCE_GPU_PHASES"] = "true"
os.environ["PYTHONPATH"] = PROJECT_ROOT


def find_spark_home():
    """Find Spark installation."""
    # Check if pyspark is installed (it bundles Spark)
    try:
        import pyspark
        return os.path.dirname(pyspark.__file__)
    except ImportError:
        pass

    # Check local download
    local_spark = os.path.join(PROJECT_ROOT, "cluster", "native", "spark", "spark-3.5.1-bin-hadoop3")
    if os.path.exists(local_spark):
        return local_spark

    print("ERROR: Spark not found. Install pyspark: python -m pip install pyspark")
    sys.exit(1)


def start_master(spark_home):
    """Start Spark master on this host."""
    print(f"Starting Spark Master on {MASTER_IP}:{MASTER_PORT}...")

    # Use pyspark's built-in master via SparkSession
    # (avoids needing to find/run spark-class.cmd)
    from pyspark.sql import SparkSession

    master_spark = (
        SparkSession.builder
        .master(f"spark://{MASTER_IP}:{MASTER_PORT}")
        .appName("ClusterMaster")
        .config("spark.driver.host", MASTER_IP)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.port", str(DRIVER_PORT))
        .config("spark.driver.blockManager.port", "33005")
        .getOrCreate()
    )

    # Actually, for standalone mode we need the master process.
    # Let's start it via subprocess using pyspark's bundled scripts
    master_cmd = [
        sys.executable, "-c",
        f"""
import subprocess, os, sys
import pyspark
spark_home = os.path.dirname(pyspark.__file__)
master_class = "org.apache.spark.deploy.master.Master"
jars = os.path.join(spark_home, "jars", "*")
cmd = ["java", "-cp", jars, master_class,
       "--host", "{MASTER_IP}", "--port", "{MASTER_PORT}", "--webui-port", "8080"]
print(f"Starting: {{' '.join(cmd)}}")
subprocess.run(cmd)
"""
    ]
    return master_spark


def wait_for_workers(spark_context, min_workers=1, timeout=120):
    """Wait for at least min_workers to connect."""
    print(f"Waiting for at least {min_workers} worker(s) to connect...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            # Check executor count
            status = spark_context.statusTracker()
            executors = status.getExecutorInfos()
            # Subtract 1 for driver
            worker_count = len(executors) - 1 if executors else 0
            if worker_count >= min_workers:
                print(f"  {worker_count} worker(s) connected!")
                return True
        except Exception:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start)
        print(f"  Waiting... ({elapsed}s / {timeout}s)")

    print(f"WARNING: Timeout waiting for workers. Proceeding anyway.")
    return False


def main():
    print("=" * 60)
    print("HYBRID CLUSTER BENCHMARK")
    print("=" * 60)
    print(f"Master IP:    {MASTER_IP}")
    print(f"Spark Master: spark://{MASTER_IP}:{MASTER_PORT}")
    print(f"Driver Port:  {DRIVER_PORT}")
    print(f"Output:       {os.environ['BENCHMARK_OUTPUT_DIR']}")
    print("=" * 60)
    print()
    print("Make sure workers are running on other machines:")
    print("  docker compose -f docker-compose.worker.yml up")
    print()

    # Option 1: Use local[*] mode as Spark master (simpler, avoids standalone master)
    # This works if workers connect via the cluster manager embedded in the driver.
    # BUT for true standalone cluster, we need workers to register with a master.

    # For simplicity: run the benchmark using pyspark with standalone master.
    # The SparkSession in cluster_benchmark.py will start the master implicitly
    # when connecting to spark://MASTER_IP:7077 IF a master is running.

    # Check if master is already running (from previous start_master.bat or Docker)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    master_running = sock.connect_ex((MASTER_IP, MASTER_PORT)) == 0
    sock.close()

    if not master_running:
        print(f"No master detected at {MASTER_IP}:{MASTER_PORT}")
        print("Starting embedded master via PySpark...")
        print()
        # Fall back to local mode with all cores for this machine
        # Workers can still connect if master is started separately
        os.environ["SPARK_MASTER"] = "local[*]"
        print("NOTE: Running in local[*] mode. For true cluster:")
        print("  1. Start master: cluster\\native\\start_master.bat")
        print("  2. Start workers on other machines")
        print("  3. Re-run this script")
        print()
    else:
        print(f"Master detected at {MASTER_IP}:{MASTER_PORT} ✓")
        print()

    # Run the benchmark
    print("Starting 3-Phase Benchmark...")
    print("-" * 60)

    from pytorch_benchmark.cluster_benchmark import run_all_phases
    results = run_all_phases()

    # Generate report
    print("\nGenerating report...")
    from pytorch_benchmark.generate_cluster_report import generate_markdown_report, generate_charts, load_results

    report_path = os.path.join(os.environ["BENCHMARK_OUTPUT_DIR"], "CLUSTER_BENCHMARK_REPORT.md")
    generate_markdown_report(results, report_path)

    try:
        generate_charts(results)
    except Exception as e:
        print(f"Chart generation failed (non-critical): {e}")

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
