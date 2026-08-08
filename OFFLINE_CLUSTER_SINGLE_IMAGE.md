# Running the Cluster Offline with One Docker Image

How to bring up the whole single-box Spark cluster demo airgapped, using
only `pytorch-spark-worker:gpu`, and how to run each benchmark variant
against it from the CLI.

This is the minimal path — see
[`AIRGAP_PACKAGE_ASSESSMENT.md`](AIRGAP_PACKAGE_ASSESSMENT.md) for the full
package manifest and [`TESTING_BEFORE_AIRGAP.md`](TESTING_BEFORE_AIRGAP.md)
for the full pre-transfer validation sequence. This doc only covers: one
image, offline, cluster up, benchmarks run.

---

## 1. Why one image is enough

Root [`docker-compose.yml`](docker-compose.yml)'s cluster-mode services —
`spark-master`, `spark-worker-1`, `spark-worker-2`, and `benchmark-cluster`
(the driver) — all reference `image: pytorch-spark-worker:gpu` directly, no
`build:` step. One image plays all four roles because it's built from
[`Dockerfile.worker`](Dockerfile.worker), which bundles Java 17 + Spark
4.2.0 + GPU torch + the benchmark code together — the master role just
never touches torch, it only runs the Spark JVM.

This is the **single-machine cluster demo** (multiple containers on one
box), not the real multi-LAN-machine topology in
`cluster/docker-compose.master.yml` / `worker.yml` — see the note at the
bottom if that's what you're actually deploying.

**Requirement:** the host needs a working NVIDIA GPU + driver with Docker
GPU passthrough enabled (Docker Desktop Settings → Resources → GPU, or
NVIDIA Container Toolkit on Linux). `spark-worker-1`/`spark-worker-2`/
`benchmark-cluster` all request a GPU device reservation; without GPU
passthrough available, those containers fail to start.

---

## 2. Get the image onto the airgapped machine

**On the internet-connected build machine:**
```cmd
docker build --file Dockerfile.worker --target gpu --tag pytorch-spark-worker:gpu .
docker save pytorch-spark-worker:gpu | gzip > pytorch-spark-worker-gpu.tar.gz
```
(This is the same output `airgap\save_docker_images.bat` produces as part
of `gpu-images-combined.tar.gz` — if you already ran that script, you can
skip straight to transferring the existing file instead of rebuilding.)

Transfer `pytorch-spark-worker-gpu.tar.gz` to the airgapped machine (USB /
file share / DVD), along with the project source
(`spark_bench_mark_poc.zip` or equivalent).

**On the airgapped machine:**
```cmd
docker load -i pytorch-spark-worker-gpu.tar.gz
docker images
REM Confirm pytorch-spark-worker:gpu is listed
```

---

## 3. Bring the cluster up offline

From the project root on the airgapped machine:
```cmd
docker compose up -d spark-master spark-worker-1 spark-worker-2
```
- `image:` references only, no `build:` — `docker compose` will not attempt
  to pull or build, it uses the loaded local image directly.
- Wait for both workers to register: open `http://localhost:8080` — the
  Spark master UI should list 2 workers under "Workers."
- No `--network none` is needed for the actual run (the containers do need
  to talk to each other over Docker's default bridge network); the
  `--network none` mode is only used by `airgap\simulate_airgap_test.bat`
  to *prove* offline-readiness beforehand, not for the real run.

Check logs if a worker doesn't register:
```cmd
docker compose logs spark-worker-1
docker compose logs spark-master
```

---

## 4. Run benchmark variants from the CLI

The `benchmark-cluster` service already wires up the right environment
variables to talk to `spark-master:7077` — the cleanest way to run a
variant is `docker compose run` (overriding the entrypoint/command), so
results still land in `.\benchmark_results\` on the host via the existing
volume mount.

### 4.1 Full cluster benchmark — `cluster_benchmark.py`
Env-var driven only, no CLI flags. This is what `docker compose up
benchmark-cluster` runs by default:
```cmd
docker compose run --rm benchmark-cluster
```
Configurable via environment (already set in `docker-compose.yml`, override
with `-e` to change):

| Var | Default | Meaning |
|---|---|---|
| `BENCHMARK_SAMPLES` | 200 | inference samples per model |
| `BENCHMARK_BATCH_SIZE` | 64 | batch size |
| `BENCHMARK_PARTITIONS` | 4 | Spark data partitions |
| `BENCHMARK_MODELS` | `resnet50,mobilenet_v3,efficientnet_b0,distilbert,tabular_deep` | comma-separated model list |
| `FORCE_GPU_PHASES` | true | run GPU phases even if `torch.cuda.is_available()` misreports |

Example — fewer samples, resnet50 + distilbert only:
```cmd
docker compose run --rm -e BENCHMARK_SAMPLES=50 -e BENCHMARK_MODELS=resnet50,distilbert benchmark-cluster
```

### 4.2 Low-RPC variant — `cluster_benchmark_low_rpc.py`
Same env-var config as above (`SPARK_MASTER`, `BENCHMARK_SAMPLES`, etc.) —
workers load model + data locally and only send small metrics back over
RPC, instead of shipping full results through Spark's block transfer. Not
wired into `docker-compose.yml` as its own service; run it as a module
override on the same image:
```cmd
docker compose run --rm --entrypoint python3 benchmark-cluster -m pytorch_benchmark.cluster_benchmark_low_rpc
```

### 4.3 Inference-only, single model — `run_inference_only.py`
Has real CLI flags (unlike the two above). Fastest way to sanity-check one
model end-to-end without a full training run:
```cmd
docker compose run --rm --entrypoint python3 benchmark-cluster \
  -m pytorch_benchmark.run_inference_only --model resnet50 --samples 200
```
Flags:

| Flag | Meaning |
|---|---|
| `--model {resnet50,mobilenet_v3,efficientnet_b0,distilbert,tabular_deep}` | run one model |
| `--all` | run all models sequentially |
| `--samples N` | default 200 |
| `--batch-size N` | default 32 |
| `--seed N` | default 42 |
| `--no-gpu` | skip GPU modes |
| `--no-spark` | skip Spark modes |
| `--cpu-only` | only `torch_cpu` mode |

### 4.4 Main suite (training + pretrained inference) — `main.py`
Same image, different module — this is what the standalone
`pytorch-benchmark:cpu`/`:gpu` images run as their entrypoint, but nothing
stops you running it against the worker image too, in local (non-cluster)
mode:
```cmd
docker compose run --rm --entrypoint python3 benchmark-cluster \
  -m pytorch_benchmark.main --modes torch_gpu spark_gpu --epochs 3 --pretrained-only --model resnet50
```
Key flags:

| Flag | Meaning |
|---|---|
| `--modes {torch_cpu,torch_gpu,spark_cpu,spark_gpu,all}...` | which execution modes to benchmark |
| `--epochs N`, `--batch-size N`, `--lr F`, `--seed N` | training params |
| `--cpu-only` | quick sanity check, torch_cpu only |
| `--skip-pretrained` | skip pretrained inference phase |
| `--pretrained-only` | skip training, run only pretrained inference |
| `--models {resnet50,mobilenet_v3,efficientnet_b0,distilbert,tabular_deep}...` | subset of pretrained models |
| `--model {...}` | single model through inference across all modes |
| `--data-type {structured,unstructured,both}` | restrict data type |

---

## 5. The one gap this doesn't cover

Any variant that touches `resnet50` / `mobilenet_v3` / `efficientnet_b0`
loads ImageNet weights from `download.pytorch.org` on first use inside the
container — see
[`AIRGAP_PACKAGE_ASSESSMENT.md §2.1`](AIRGAP_PACKAGE_ASSESSMENT.md#21-pretrained-imagenet-weights-are-not-bundled-breaks-mid-run).
With no network, that call hangs/fails mid-benchmark. Before relying on
this offline, either:
- bake the Torch hub cache (`~/.cache/torch/hub/checkpoints/`) into the
  image at build time (`COPY` it in `Dockerfile.worker` before the final
  stage), or
- mount a pre-populated cache directory into the container at runtime:
  ```cmd
  docker compose run --rm -v C:\path\to\torch_hub_cache:/root/.cache/torch/hub benchmark-cluster
  ```
`distilbert` and `tabular_deep` are unaffected — both are trained from
random init in this repo, no external weights fetched.

---

## 6. If you actually need the multi-machine (not single-box) topology

Everything above is `docker-compose.yml`'s single-box demo cluster. If
you're deploying to separate physical/VM machines instead (matching
`cluster/docker-compose.master.yml` + `cluster/docker-compose.worker.yml`),
note that compose file has no `image:` tag — it builds locally and has no
default `target:`, which resolves to the **last** stage in
`Dockerfile.worker` (`cpu`, not `gpu`). For that path to use the same
offline `pytorch-spark-worker:gpu` image loaded above, add to the worker
service in that compose file:
```yaml
services:
  spark-worker:
    image: pytorch-spark-worker:gpu   # add this — skips build, uses loaded image
    # build: ...                       # remove or comment out on the airgapped target
```
and run `docker compose -f cluster/docker-compose.worker.yml up --no-build`.
