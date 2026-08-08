# Testing Native + Docker Before Shipping Airgapped

Practical runbook for validating this repo on a connected Windows dev box
before packaging it for an airgapped target. This restates and confirms the
existing [`airgap/TESTING.md`](airgap/TESTING.md) — that file is accurate
and current (all scripts it references exist:
[`rebuild.bat`](rebuild.bat)/[`rebuild.sh`](rebuild.sh),
[`test_native.bat`](test_native.bat), [`test_docker.bat`](test_docker.bat),
[`docker-compose.yml`](docker-compose.yml)). Use `airgap/TESTING.md` as the
source of truth; this file is the condensed sequence plus one bug found
while re-checking it.

---

## 0. GPU/driver check (skip if CPU-only)

```cmd
nvidia-smi
REM "CUDA Version:" field in the header must read 12.8 or higher

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REM Expect: 2.11.0+cu128 True <your GPU name>
```
Pinned wheels need driver ≥570.65 (Windows) / ≥570.26 (Linux). RTX 30/40/50
series all satisfy this. No GPU or an older driver just means the CPU image
is what you test — not a blocker.

## 1. Test sequence — each phase gates the next

| Phase | Command | Proves |
|---|---|---|
| A — Rebuild | `rebuild.bat` (or `rebuild.bat -docker-only`) | All 4 Docker images + native env build clean from the pinned versions (torch 2.11.0+cu128, pyspark 4.2.0, Python 3.12) |
| B — GPU preflight | `cluster\native\check_gpu.py` | `torch.cuda.is_available()` is True on the *exact* interpreter Spark launches tasks with (mismatched interpreter is the #1 cause of silent CPU fallback) |
| C — Native single-node | `test_native.bat` | Native Python 3.12 + pyspark 4.2.0 + GPU torch work together in `local[*]` mode, no master/worker needed |
| D — Docker single-container | `docker compose up benchmark-quick` / `benchmark-cpu` / `benchmark-gpu` (or `make run-quick` / `run-cpu` / `run-gpu`) | Each image runs standalone before testing the multi-container cluster |
| E — Docker multi-container cluster | `test_docker.bat` | `spark-master` + 2 workers register as containers on one box (check `http://localhost:8080` shows both workers) — closest single-machine proxy for the real distributed topology |
| F — **Airgap offline simulation** | `airgap\simulate_airgap_test.bat` | The actual pre-transfer gate — see below |
| G — Package for transfer | `airgap\download_all.bat` → `airgap\save_docker_images.bat` → `airgap\split_for_dvd.bat` | Only run once F is all-green |

Don't skip ahead — a failure at any phase is the same failure you'd hit on
the airgapped machine, just with internet available to fix it here.

## 2. Phase F in detail — what it actually checks

`airgap\simulate_airgap_test.bat` fakes the airgapped environment on your
connected box: pip runs with `--no-index`, Docker runs with `--network
none`, Spark runs `local[*]` so no real cluster is needed. 10 checks:

| # | Environment | Checks |
|---|---|---|
| A1 | Native | torch import + CUDA GPU detection |
| A2 | Native | pyspark import |
| A3 | Native | Spark local session (`local[2]`, RDD sum) |
| A4 | Native | GPU tensor matmul |
| A5 | Native | `pytorch_benchmark` module importable |
| B1 | Docker | `pytorch-benchmark:cpu` image present |
| B2 | Docker | `pytorch-spark-worker:gpu` image present |
| B3 | Docker | CPU container imports work with `--network none` |
| B4 | Docker | GPU worker CUDA works with `--network none --gpus all` |
| B5 | Docker | Spark local session inside container with `--network none` |

Log: `benchmark_results\airgap_test\airgap_sim_<timestamp>.log`. All 10 must
pass before Phase G.

**Known gap: A1–A5 don't cover pretrained-weight loading.** They check
imports and a bare GPU matmul, not an actual `load_pretrained_model(...)`
call. `resnet50`/`mobilenet_v3`/`efficientnet_b0` download ImageNet weights
from `download.pytorch.org` on first use — see
[`AIRGAP_PACKAGE_ASSESSMENT.md`](AIRGAP_PACKAGE_ASSESSMENT.md#21-pretrained-imagenet-weights-are-not-bundled-breaks-mid-run)
for the fix (cache the weights, add an A6 check). Without that, this suite
reports 10/10 pass and the pretrained-model phase of the benchmark still
fails offline.

## 3. Bug found while verifying this doc

[`airgap\simulate_airgap_test.bat:18`](airgap/simulate_airgap_test.bat#L18)
hardcodes:
```
set PYTHON=C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe
```
Same issue already flagged for `download_all.bat` and `save_docker_images.bat`
in `AIRGAP_PACKAGE_ASSESSMENT.md` §2.2 — breaks Phase F on any machine other
than the original author's. `install_native.bat` has working `py -3.12`
detection logic; copy it into all three scripts.

## 4. Later: testing on Linux (production target)

No native path on Linux — Docker only, by design (see
[`airgap/ARCHITECTURE.md`](airgap/ARCHITECTURE.md)).

```bash
./rebuild.sh                                    # or: docker load < gpu-images-combined.tar
sudo bash container_toolkit/install_offline.sh  # NVIDIA Container Toolkit — Docker Engine needs this, Docker Desktop doesn't
./airgap/simulate_airgap_test.sh                # same B1-B5 checks, ported to bash; B4 skips (not fails) if nvidia-smi is absent
```
Then the multi-host cluster test once a second Linux box exists:
`docker compose -f cluster/docker-compose.master.yml up` /
`docker-compose.worker.yml up` on separate hosts.

The Windows Docker phases (D, E, F's B-checks) transfer directly — same
image, same behavior, new host. The native-only checks (B, C, A1/A2/A3/A5)
don't carry over to Linux; there's nothing native to re-validate there.
