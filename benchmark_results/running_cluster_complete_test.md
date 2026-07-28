set SPARK_MASTER=spark://192.168.4.100:7077
set SPARK_DRIVER_HOST=192.168.4.100
set SPARK_DRIVER_PORT=33000
set FORCE_GPU_PHASES=true
set BENCHMARK_MODELS=resnet50,mobilenet_v3,efficientnet_b0,distilbert,tabular_deep
set BENCHMARK_SAMPLES=200
set BENCHMARK_PARTITIONS=4
python -m pytorch_benchmark.cluster_benchmark_low_rpc
