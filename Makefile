# ===========================================================================
# Makefile for PyTorch Benchmark Suite
# ===========================================================================

.PHONY: help build-cpu build-gpu run-cpu run-gpu run-quick run-cluster clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build-cpu: ## Build CPU Docker image
	docker build --target cpu -t pytorch-benchmark:cpu .

build-gpu: ## Build GPU Docker image (requires NVIDIA base image)
	docker build --target gpu -t pytorch-benchmark:gpu .

build-worker-cpu: ## Build CPU Spark worker image
	docker build --file Dockerfile.worker --target cpu -t pytorch-spark-worker:cpu .

build-worker-gpu: ## Build GPU Spark worker image
	docker build --file Dockerfile.worker --target gpu -t pytorch-spark-worker:gpu .

rebuild: build-cpu build-gpu build-worker-cpu build-worker-gpu ## Rebuild all 4 Docker images (cache-aware — see rebuild.sh for native env too)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run-quick: ## Run quick sanity check (CPU, 2 epochs)
	docker compose up benchmark-quick

run-cpu: ## Run CPU benchmark (torch_cpu + spark_cpu)
	docker compose up benchmark-cpu

run-gpu: ## Run full benchmark with GPU (all 4 modes)
	docker compose up benchmark-gpu

run-cluster: ## Run with Spark cluster (multi-container)
	docker compose up spark-master spark-worker benchmark-cluster

# ---------------------------------------------------------------------------
# Custom runs
# ---------------------------------------------------------------------------

run-custom: ## Run with custom args: make run-custom ARGS="--epochs 10 --batch-size 128"
	docker compose run --rm benchmark-cpu $(ARGS)

run-custom-gpu: ## Run GPU with custom args: make run-custom-gpu ARGS="--epochs 10"
	docker compose run --rm benchmark-gpu $(ARGS)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

logs: ## Show logs from last run
	docker compose logs

results: ## List benchmark results
	@ls -la benchmark_results/ 2>/dev/null || echo "No results yet. Run a benchmark first."

clean: ## Remove containers and output
	docker compose down -v --remove-orphans
	rm -rf benchmark_results/*.json benchmark_results/*.txt

clean-all: ## Remove everything including images
	docker compose down -v --remove-orphans --rmi all
	rm -rf benchmark_results/

shell-cpu: ## Open shell in CPU container
	docker compose run --rm --entrypoint /bin/bash benchmark-cpu

shell-gpu: ## Open shell in GPU container
	docker compose run --rm --entrypoint /bin/bash benchmark-gpu
