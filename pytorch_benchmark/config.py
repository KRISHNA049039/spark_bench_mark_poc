"""
Global configuration for the benchmark suite.
"""
import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
STRUCTURED_NUM_SAMPLES = 10_000
STRUCTURED_NUM_FEATURES = 20
STRUCTURED_NUM_CLASSES = 5

UNSTRUCTURED_NUM_SAMPLES = 2_000
UNSTRUCTURED_IMAGE_SIZE = (1, 28, 28)  # grayscale 28x28 (MNIST-like)
UNSTRUCTURED_NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Spark settings
# ---------------------------------------------------------------------------
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
SPARK_EXECUTOR_CORES = int(os.getenv("SPARK_EXECUTOR_CORES", "2"))
SPARK_NUM_EXECUTORS = int(os.getenv("SPARK_NUM_EXECUTORS", "2"))

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
MONITOR_INTERVAL_SEC = 0.5  # sampling interval for resource monitor

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.getenv("BENCHMARK_OUTPUT_DIR", "benchmark_results")

# ---------------------------------------------------------------------------
# Reproducibility tolerance
# ---------------------------------------------------------------------------
RESULT_ATOL = 1e-5  # absolute tolerance for numerical comparison
RESULT_RTOL = 1e-4  # relative tolerance
