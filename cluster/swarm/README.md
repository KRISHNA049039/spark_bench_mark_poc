# Docker Swarm Cluster Setup

Docker Swarm creates an **overlay network** that spans all machines. All containers see each other by hostname — no IP mismatch, no port mapping issues.

## Why Swarm Solves Yesterday's Problems

| Problem | Docker Compose (yesterday) | Docker Swarm (today) |
|---------|---------------------------|---------------------|
| Driver IP | `172.x.x.x` (VM internal) | `benchmark-driver` (resolvable by all) |
| Worker IP | `172.x.x.x` (different VM) | `spark-worker.1.xxx` (resolvable) |
| Block transfer | ❌ Can't reach `172.x.x.x` across machines | ✅ Overlay routes internally |
| Port access | Only mapped ports | All ports on overlay network |
| Service discovery | Manual IPs | Automatic by service name |

## Prerequisites

- Docker Desktop installed on ALL machines (you already have this)
- All machines on same LAN (192.168.4.x — confirmed)
- Port 2377, 7946, 4789 open between machines (Docker Swarm ports)

## Setup Steps

### Step 1: Initialize Swarm on Master (192.168.4.100)

```cmd
docker swarm init --advertise-addr 192.168.4.100
```

This prints a join token. Copy it.

### Step 2: Join Workers to Swarm

On **Worker 1** (192.168.4.101):
```cmd
docker swarm join --token <paste-token-here> 192.168.4.100:2377
```

On **Worker 2** (192.168.4.102):
```cmd
docker swarm join --token <paste-token-here> 192.168.4.100:2377
```

### Step 3: Verify Swarm (on master)

```cmd
docker node ls
```

Should show 3 nodes: 1 manager (Leader) + 2 workers.

### Step 4: Build and Push Image

The swarm needs the image available on ALL nodes. Two options:

**Option A: Build on each machine (simplest):**

On ALL 3 machines:
```cmd
cd spark_bench_mark_poc
docker build -f Dockerfile.worker -t pytorch-spark-worker:latest .
```

**Option B: Use a registry (production approach):**
```cmd
REM On master:
docker build -f Dockerfile.worker -t 192.168.4.100:5000/pytorch-spark-worker:latest .
docker push 192.168.4.100:5000/pytorch-spark-worker:latest
```

### Step 5: Deploy the Stack

On master:
```cmd
cd spark_bench_mark_poc/cluster/swarm
docker stack deploy -c docker-compose.swarm.yml spark
```

### Step 6: Monitor

```cmd
REM Check services
docker stack services spark

REM Check where containers landed
docker stack ps spark

REM Spark UI
start http://localhost:8080

REM Driver logs
docker service logs spark_benchmark-driver -f

REM Worker logs
docker service logs spark_spark-worker -f
```

### Step 7: Get Results

```cmd
REM Find the driver container
docker ps | findstr benchmark-driver

REM Copy results out
docker cp <container-id>:/app/benchmark_results ./benchmark_results
```

Or use a bind mount (see below).

## Teardown

```cmd
docker stack rm spark
```

To leave swarm (on workers):
```cmd
docker swarm leave
```

To disband swarm (on master):
```cmd
docker swarm leave --force
```

## Running Different Models

Edit `BENCHMARK_MODELS` in `docker-compose.swarm.yml`:

```yaml
- BENCHMARK_MODELS=resnet50           # single model
- BENCHMARK_MODELS=resnet50,distilbert # multiple
- BENCHMARK_MODELS=resnet50,mobilenet_v3,efficientnet_b0,distilbert,tabular_deep  # all
```

Then redeploy:
```cmd
docker stack deploy -c docker-compose.swarm.yml spark
```

## Scaling Workers

Add more machines to swarm, then scale:
```cmd
docker service scale spark_spark-worker=4
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No such image` on workers | Build image on ALL machines before deploying |
| Workers can't reach master | Check firewall: ports 2377, 7946, 4789 |
| `network not found` | Wait 10s and retry deploy |
| Results not appearing | Check driver logs: `docker service logs spark_benchmark-driver` |
| Want to restart just driver | `docker service update --force spark_benchmark-driver` |
