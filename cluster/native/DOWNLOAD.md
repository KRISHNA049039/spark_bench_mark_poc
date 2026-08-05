# Downloading the Native (Non-Docker) Packages — Windows & Linux

Steps to get the pinned Python environment — `torch==2.11.0` (`+cpu` or
`+cu128`), `torchvision==0.26.0`, `pyspark==4.2.0`, and the rest of
`pytorch_benchmark/requirements-base.txt` — installed natively (no Docker)
on a Windows or Linux machine **with internet access**.

> **Scope note:** this is the *online* native install, for dev/test boxes.
> It is not the same thing as the airgap kit's offline wheel bundle (that's
> `airgap/download_all.bat`, covered in [airgap/README.md](../../airgap/README.md)
> and [BUILD_AND_SHIP.md](../../BUILD_AND_SHIP.md) §4). And it's not the
> production Linux deployment path either — an *airgapped* Linux target
> ships as Docker images only, by deliberate design (see
> [airgap/ARCHITECTURE.md](../../airgap/ARCHITECTURE.md)). This doc is for
> a Linux or Windows machine that has internet access right now and you
> just want the native packages installed on it.

Why the version matters at all: torch/torchvision/pyspark are pinned to
**exact** versions in `pytorch_benchmark/requirements-*.txt` so native and
Docker builds land on identical package versions — see
[BUILD_AND_SHIP.md](../../BUILD_AND_SHIP.md) §6 for why that's the whole
reproducibility guarantee. Installing from a `>=` range or `pip install
torch` with no pin defeats that.

---

## Windows

**1. Install Python 3.12** (if not already present)
Download from https://www.python.org/downloads/, check "Add Python to
PATH" during install. Confirm:
```cmd
python --version
REM Expect: Python 3.12.x
```
If this prints 3.14 or another version, you likely have the Windows Store
alias resolving first — install 3.12 from python.org and make sure it's
ahead of the Store alias on `PATH`, or call the interpreter by its full
path (e.g. `C:\Users\<you>\AppData\Local\Python\pythoncore-3.12-64\python.exe`).

**2. Install Java 17** (needed by Spark; pyspark does not bundle a JVM)
Download from https://adoptium.net/temurin/releases/?version=17, check
"Set JAVA_HOME" during install.

**3. Install the pinned packages**

GPU machine:
```cmd
cluster\native\install_gpu_worker.bat
```
Installs `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` from
`pytorch_benchmark\requirements-torch-gpu.txt`, then everything in
`requirements-base.txt` (pyspark, numpy, pandas, scikit-learn, etc.).

CPU-only machine (e.g. a driver/master node with no GPU):
```cmd
pip install -r pytorch_benchmark\requirements-base.txt -r pytorch_benchmark\requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
```

**4. Verify**
```cmd
python cluster\native\check_gpu.py
```
Expect `torch=2.11.0+cu128` and `CUDA=True` on a GPU box (or
`torch=2.11.0+cpu` with a CPU-only warning on a CPU box — not a failure).

No separate Spark download needed: `pyspark` bundles Spark's binaries
inside the pip package. `start_master.bat`/`start_worker.bat`/
`run_benchmark.bat` derive `SPARK_HOME` from it directly
(`import pyspark, os; os.path.dirname(pyspark.__file__)`).

---

## Linux

**1. Install Python 3.12**

Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip
```
RHEL/Fedora:
```bash
sudo dnf install -y python3.12
```
If your distro's default repos don't have 3.12 yet, use the
[deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)
(Ubuntu) or build from python.org source. Confirm:
```bash
python3.12 --version
```

**2. Install Java 17** (needed by Spark; pyspark does not bundle a JVM)

Debian/Ubuntu:
```bash
sudo apt-get install -y openjdk-17-jre-headless
```
RHEL/Fedora:
```bash
sudo dnf install -y java-17-openjdk-headless
```
Confirm `JAVA_HOME` is set (or exported in your shell profile) — pyspark
looks for it the same way on Linux as on Windows.

**3. Install the pinned packages**

GPU machine:
```bash
PYTHON=python3.12 ./cluster/native/install_gpu_worker.sh
```
(Or just `./cluster/native/install_gpu_worker.sh` if `python3` on your
`PATH` already resolves to 3.12.) Same pins as Windows:
`torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` from
`requirements-torch-gpu.txt`, then `requirements-base.txt`.

CPU-only machine:
```bash
python3.12 -m pip install -r pytorch_benchmark/requirements-base.txt -r pytorch_benchmark/requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
```

**4. Verify**
```bash
python3.12 cluster/native/check_gpu.py
```
Same expected output as Windows (`check_gpu.py` is plain Python, no
OS-specific code).

---

## What's actually different between the two OSes

| | Windows | Linux |
|---|---|---|
| Torch/torchvision wheel | `win_amd64` | `manylinux_2_28_x86_64` |
| Java source | Adoptium installer, or the portable JRE zip in the airgap kit | Distro package (`apt`/`dnf`) |
| Spark | Bundled inside `pyspark`, same on both | Bundled inside `pyspark`, same on both |
| Multi-node cluster scripts (`start_master.bat`/`start_worker.bat`/`run_benchmark.bat`) | Provided | **Not provided** — these are Windows batch files with no Linux port. A Linux native multi-node cluster would need the equivalent Spark master/worker commands run by hand (`$SPARK_HOME/sbin/start-master.sh` etc., using the same `SPARK_HOME` derivation as above) |

The last row is the one gap worth knowing about: single-machine native
install and `check_gpu.py` work identically on both OSes, but the
scripted multi-node topology (`cluster/native/*.bat`) is Windows-only.
Linux's supported multi-node path is the Docker Compose cluster
(`docker-compose.yml`'s `spark-master`/`spark-worker-*`/`benchmark-cluster`
services) — see [BUILD_AND_SHIP.md](../../BUILD_AND_SHIP.md) §1.

---

## Running this in an airgapped environment

Everything above assumes internet access at install time. Here's how each
OS's story changes once the target machine has none.

### Windows — this maps directly onto the existing airgap kit

The commands above are the *online* version of exactly what
`airgap/download_all.bat` + `airgap/install_native.bat` already automate
for offline transfer. Same pinned files, same packages — just split across
two machines instead of run in one step:

| | Online (this doc) | Airgapped (existing scripts) |
|---|---|---|
| Fetch | `pip install --index-url .../cu128 -r requirements-torch-gpu.txt` — downloads *and* installs in one step, needs internet | `pip download --dest wheels\ --index-url .../cu128 -r requirements-torch-gpu.txt` on an internet machine — downloads only, doesn't install |
| Transfer | — | Copy `airgap\packages\` to the airgapped machine (DVD/USB/network share) |
| Install | (already done by the fetch step) | `pip install --no-index --find-links wheels\ -r requirements-torch-gpu.txt -r requirements-base.txt` on the airgapped machine — installs from the local folder only, zero network calls |

In practice, run `airgap\download_all.bat` on an internet-connected
Windows machine (produces `airgap\packages\native\wheels\`, plus Spark and
the portable Java JRE), transfer `airgap\packages\` to the airgapped
machine, then run `airgap\install_native.bat` there — it extracts Java and
Spark and does the `--no-index` pip install shown above automatically.
Full walkthrough: [airgap/README.md](../../airgap/README.md). Test it
still works before you rely on it:
[airgap/TESTING.md](../../airgap/TESTING.md) Phase F
(`airgap\simulate_airgap_test.bat`, checks A1-A5 for exactly this native
path).

#### Getting Python 3.12 itself onto the airgapped machine

All of the above assumes Python 3.12 is **already installed** on the
airgapped machine — `install_native.bat` only installs *packages* into an
existing interpreter via `pip install --no-index`, the same way
`airgap/download_all.bat`'s wheel bundle only carries packages, not Python
itself. Wheels are built *for* a Python install; they can't create one.

If this machine has the Python Install Manager (`py` — check with `py
--version`, this is what `py install 3.12` used earlier in this guide), it
has a built-in offline-prep mode for exactly this, verified working:

**1. On the internet-connected machine**, prepare an offline install package:
```cmd
py install --download="D:\spark_pytorch_poc\airgap\packages\native\python312" 3.12
```
Produces `index.json` + `pythoncore-3.12-64-3.12.10.zip` (~32MB total) in
that folder — small enough to just ride along with the wheels bundle on
whatever media you're already using.

**2. Transfer** the `python312\` folder to the airgapped machine, same way
as `airgap\packages\` (USB / network share / disc).

**3. On the airgapped machine**, install Python 3.12 from that local folder
— no network access required:
```cmd
py install -s "D:\path\to\python312" 3.12
```

**4. Then** proceed with the wheel install exactly as in the Windows table
above:
```cmd
py -3.12 -m pip install --no-index --find-links "D:\path\to\wheels" -r pytorch_benchmark\requirements-torch-gpu.txt -r pytorch_benchmark\requirements-base.txt
```

If the airgapped machine doesn't have the `py` launcher at all (older
Windows setup, or the Install Manager was never set up there), this
mechanism doesn't apply — fall back to carrying a classic python.org
`.exe` installer for Python 3.12 on the transfer media instead, and run
that directly.

> This offline-Python-install step is **not yet wired into
> `download_all.bat`/`install_native.bat`** — it's a manual addition on
> top of the existing scripts for now, not something those scripts do for
> you automatically.

### Linux — no scripted offline path; Docker is the shipped route

As the scope note at the top of this doc says, an airgapped **Linux**
target ships as Docker images only (see
[airgap/ARCHITECTURE.md](../../airgap/ARCHITECTURE.md)) — there's no
`download_all.sh`/`install_native.sh` in this repo. That's a deliberate
scoping call, not an oversight: the Docker path already covers Linux, so a
second offline-delivery mechanism for the same result wasn't worth
building and maintaining. `airgap/simulate_airgap_test.sh` is the
supported way to verify a Linux airgapped machine — Docker-only, no native
checks.

If you specifically need a **native** (non-Docker) install on an
airgapped Linux box anyway, the same two-step pattern applies — it's just
not wrapped in a script here:

```bash
# On a Linux machine WITH internet access — pip download defaults to the
# platform it's running on, so this naturally produces manylinux wheels:
mkdir -p wheels
python3.12 -m pip download --dest wheels --index-url https://download.pytorch.org/whl/cu128 -r pytorch_benchmark/requirements-torch-gpu.txt
python3.12 -m pip download --dest wheels -r pytorch_benchmark/requirements-base.txt

# Transfer wheels/ to the airgapped Linux machine (USB/network share), then:
python3.12 -m pip install --no-index --find-links wheels -r pytorch_benchmark/requirements-torch-gpu.txt -r pytorch_benchmark/requirements-base.txt
```

Same guarantees as the Windows kit: `pip download` walks the full
dependency tree so `wheels/` is self-contained, and `--no-index` on the
install step means it never touches the network. You'd also need to carry
over Java 17 and, if you want the standalone (not `pyspark`-bundled) Spark
distribution, `spark-4.2.0-bin-hadoop3.tgz` — manually, since nothing here
automates that for Linux the way `install_native.bat` does for Windows.
