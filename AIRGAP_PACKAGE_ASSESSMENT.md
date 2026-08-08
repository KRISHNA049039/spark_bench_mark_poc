# Airgap Package Assessment & Framework Review

Assessment of what's needed to run `spark_bench_mark_poc` in an airgapped
environment (native + Docker), plus a review of whether this repo can serve
as a reusable framework for future applications.

Pairs with the existing [`airgap/README.md`](airgap/README.md) and
[`airgapped_dep.md`](airgapped_dep.md) — this file records gaps found while
verifying those against the actual code (Dockerfiles, requirements files,
`.bat` scripts), not a replacement for either.

---

## 1. Package Manifest

The repo already has a working airgap kit ([`airgap/`](airgap/)) with pinned
requirements files that are the single source of truth for Docker, native,
and the wheel bundle — that part is done correctly.

### 1.1 Python wheels

From [`pytorch_benchmark/requirements-base.txt`](pytorch_benchmark/requirements-base.txt)
— identical across native and Docker:

| Package | Version |
|---|---|
| pyspark | 4.2.0 |
| numpy | 1.26.4 |
| pandas | 2.2.1 |
| scikit-learn | 1.4.2 |
| matplotlib | 3.8.4 |
| psutil | 5.9.8 |
| GPUtil | 1.4.0 |
| tabulate | 0.9.0 |

From [`requirements-torch-gpu.txt`](pytorch_benchmark/requirements-torch-gpu.txt) /
[`requirements-torch-cpu.txt`](pytorch_benchmark/requirements-torch-cpu.txt):

| Package | GPU build | CPU build |
|---|---|---|
| torch | 2.11.0+cu128 | 2.11.0+cpu |
| torchvision | 0.26.0+cu128 | 0.26.0+cpu |

Plus transitive deps pip resolves automatically: `py4j` (pyspark),
`joblib`/`scipy`/`threadpoolctl` (sklearn), `pillow` (torchvision),
`contourpy`/`fonttools`/`kiwisolver`/`cycler`/`pyparsing` (matplotlib), and
for the cu128 torch build, the bundled `nvidia-*` CUDA runtime wheels
(~2–3 GB — the bulk of the download bundle).

### 1.2 Non-pip artifacts

| Item | Native (Windows) | Docker |
|---|---|---|
| Python 3.12 installer | Needed — **not** included in the kit, get from python.org | Baked into image (`python:3.12-slim`) |
| Temurin JRE 17.0.11+9 | Auto-downloaded by `download_all.bat` | Baked into image |
| `spark-4.2.0-bin-hadoop3.tgz` | Auto-downloaded by `download_all.bat` | Baked into image |
| NVIDIA driver | Needed — **not** included in the kit | Host-level, not in image |
| NVIDIA Container Toolkit RPMs | n/a | Linux/RHEL targets only — kit handles this |
| Docker Desktop / Engine installer | n/a | Needed — **not** included in the kit |

The Docker images ([`Dockerfile`](Dockerfile), [`Dockerfile.worker`](Dockerfile.worker))
`wget` Java and Spark **at build time**, so images can only ever be built on
the internet-connected machine and shipped as `.tar`/`.tar.gz` — never build
on the airgapped target. `airgap/save_docker_images.bat` already does this
correctly.

---

## 2. Gaps Found

### 2.1 Pretrained ImageNet weights are not bundled (breaks mid-run)

[`pretrained_models.py:90-129`](pytorch_benchmark/pretrained_models.py#L90-L129)
loads `ResNet50_Weights.DEFAULT`, `MobileNet_V3_Small_Weights.DEFAULT`, and
`EfficientNet_B0_Weights.DEFAULT`. Each triggers a runtime download of a
`.pth` file from `download.pytorch.org` into the Torch hub cache
(`~/.cache/torch/hub/checkpoints/`) on first use.

Airgapped, this is a connection timeout, not a clean error — and it happens
**after** install succeeds and all 10 existing smoke tests pass, because
`simulate_airgap_test.bat` only checks imports and a GPU matmul, not an
actual pretrained-model load. This surfaces mid-benchmark instead of
pre-transfer.

**Fix:**
1. On the build machine, run each loader once (e.g.
   `load_pretrained_model("resnet50")`, `"mobilenet_v3"`,
   `"efficientnet_b0"`) to populate the Torch hub cache.
2. Ship the resulting `~/.cache/torch/hub/checkpoints/` folder (~130 MB)
   as part of the offline bundle.
3. On the target, set `TORCH_HOME` to point at the shipped cache (native),
   or `COPY` the cache folder into the Docker image at the same path
   the container's `$HOME/.cache/torch/hub/checkpoints/` resolves to.
4. Add a new smoke test (e.g. `A6`) that actually calls
   `load_pretrained_model(...)` with `--network none` / no internet, so
   this is caught before burning DVDs, not after.

### 2.2 Hardcoded build-machine Python path

[`airgap/download_all.bat:22`](airgap/download_all.bat#L22) and
[`airgap/save_docker_images.bat:14`](airgap/save_docker_images.bat#L14) hardcode:
```
C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe
```
This breaks on any build machine other than the original author's.
[`install_native.bat:32-54`](airgap/install_native.bat#L32-L54) already has
correct `py -3.12` launcher-based detection with fallback paths — that logic
should be copied into `download_all.bat` and `save_docker_images.bat`.

### 2.3 `airgapped_dep.md` is stale and contradicts the working kit

[`airgapped_dep.md`](airgapped_dep.md) documents **Python 3.14**, **Spark
4.2.0 via pip's pyspark wheel only**, and different package names throughout.
The actual working kit ([`airgap/`](airgap/)) uses **Python 3.12** end to
end, and the wheels are `cp312`-only ([`install_native.bat:59-64`](airgap/install_native.bat#L59-L64)
enforces this). Following `airgapped_dep.md` as written would produce a
non-functional bundle. Recommend either updating it to match `airgap/` or
removing it in favor of `airgap/README.md`.

### 2.4 Wheel platform mismatch risk (currently masked)

`pip download` run on Windows (as `download_all.bat` does) fetches
`win_amd64` wheels. The Linux Docker images need `manylinux` wheels instead.
This isn't a problem today because the Docker images are built with network
access on the connected machine (`RUN pip install --index-url ...` inside
the Dockerfile) rather than from the downloaded wheel folder — but it would
break silently if the build process ever switches to installing Docker
images from the same `wheels/` folder used for the native install.

---

## 3. Framework Reusability Assessment

**Short answer: the airgap kit — yes, as-is. The benchmark code — not
without refactoring first.**

### 3.1 What's reusable today

The genuinely portable asset is [`airgap/`](airgap/) plus the three-file
pinned-requirements pattern (`requirements-base.txt` /
`requirements-torch-cpu.txt` / `requirements-torch-gpu.txt` as one source of
truth consumed identically by Docker, native install, and the offline wheel
downloader). This is provider-agnostic and solves the "keep Docker and
native on identical versions" problem cleanly — worth lifting into any
future offline-deployed project unchanged.

`BaseRunner` ([`runners/base_runner.py`](pytorch_benchmark/runners/base_runner.py))
is also a clean extension point: a proper `ABC` with four working
subclasses (`torch_cpu`, `torch_gpu`, `spark_cpu`, `spark_gpu`). Adding a new
execution mode is straightforward.

### 3.2 What blocks framework use

1. **Global config, not an object.** [`config.py`](pytorch_benchmark/config.py)
   is module-level constants read from env vars at import time; every module
   does `from config import BATCH_SIZE`. This makes it impossible to run two
   configurations in the same process and blocks programmatic overrides —
   a hard ceiling for library/framework use. Needs to become a config object
   threaded through explicitly.

2. **Model dispatch is an if/elif chain, not registry-driven.**
   [`pretrained_models.py:437-450`](pytorch_benchmark/pretrained_models.py#L437-L450)
   hardcodes `if model_name == "resnet50": ... elif ...` despite
   `AVAILABLE_MODELS` already existing as a registry dict right above it.
   Adding a new model requires editing the dispatch function instead of just
   registering an entry. Storing the loader callable directly in
   `AVAILABLE_MODELS` would make this open for extension.

3. **Significant duplication between cluster variants.**
   [`cluster_benchmark.py`](pytorch_benchmark/cluster_benchmark.py) (735
   lines) and [`cluster_benchmark_low_rpc.py`](pytorch_benchmark/cluster_benchmark_low_rpc.py)
   (570 lines) are near-duplicates that have already started to diverge —
   `cluster_benchmark_low_rpc.py:223-225` calls
   `torchvision.models.resnet50(weights=...)` directly instead of going
   through `load_pretrained_model()`, which is exactly why a weights-caching
   fix (Section 2.1) would need to be applied in two places instead of one.
   [`main.py`](pytorch_benchmark/main.py) (1157 lines) also mixes CLI
   parsing, orchestration, and reporting in one file.

### 3.3 Recommendation

Treat the current benchmark code as a solid reference implementation to fork
per-project, and extract `airgap/` as the actual reusable framework piece.
Turning the benchmark half into a proper framework is a scoped effort:
convert `config.py` to a passed-in config object, make model dispatch
registry-driven, and collapse the two cluster-benchmark variants into one
parameterized implementation.

---

*Generated 2026-08-08, based on a review of `Dockerfile`, `Dockerfile.worker`,
`airgap/*.bat`, `pytorch_benchmark/requirements-*.txt`, `config.py`,
`pretrained_models.py`, `runners/base_runner.py`, `cluster_benchmark.py`,
and `cluster_benchmark_low_rpc.py`.*
