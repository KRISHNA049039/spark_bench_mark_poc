# Extending This Repo With New Models Later

Answers one question: if you add models down the line, does this codebase
act as a framework where you register a model once and it's usable
everywhere — or does it need surgery per model? Right now it's the latter.
This documents exactly what has to change today to add one model, and what
a real fix looks like.

Builds on the framework assessment in
[`AIRGAP_PACKAGE_ASSESSMENT.md §3`](AIRGAP_PACKAGE_ASSESSMENT.md#3-framework-reusability-assessment)
and the Spark/flag distribution discussion in
[`OFFLINE_CLUSTER_SINGLE_IMAGE.md §6`](OFFLINE_CLUSTER_SINGLE_IMAGE.md#6-caveat-which-script-decides-cpu-vs-gpu-differs).

---

## Short answer

**Not yet, but the shape of a real registry is already there** —
[`pretrained_models.py`](pytorch_benchmark/pretrained_models.py) has an
`AVAILABLE_MODELS` dict that looks like a proper model registry. In
practice it isn't wired up as one: adding a single new model today means
touching **six separate places by hand**, four of which just re-hardcode
the same model-name list independently and will silently drift out of sync
if you miss one. This is fixable without a large rewrite — it's a
half-day, low-risk change — but as of now it is not something you can do by
"just adding a model," despite the registry dict suggesting otherwise.

---

## What adding ONE model actually requires today

Say you want to add `"vit_b16"` (a vision transformer). Here's every place
that needs a matching edit, with evidence:

| # | File | What's there now | What you'd add |
|---|---|---|---|
| 1 | [`pretrained_models.py:40-76`](pytorch_benchmark/pretrained_models.py#L40-L76) | `AVAILABLE_MODELS` dict — metadata only (`type`, `description`, `input_size`, `num_classes`) | A new entry with the model's metadata |
| 2 | [`pretrained_models.py:437-450`](pytorch_benchmark/pretrained_models.py#L437-L450) | `if model_name == "resnet50": ... elif ...` — hardcoded if/elif dispatch, unrelated to the dict above | A new `elif model_name == "vit_b16": model = load_vit_b16(...)` branch, plus the `load_vit_b16()` function itself |
| 3 | [`main.py`](pytorch_benchmark/main.py) argparse `--models` (`choices=[...]`, ~line 706) | Hardcoded list: `["resnet50", "mobilenet_v3", "efficientnet_b0", "distilbert", "tabular_deep"]` — **not** derived from `AVAILABLE_MODELS.keys()` | Add `"vit_b16"` to this literal list |
| 4 | Same file, `--model` argparse (~line 713) | Same hardcoded list, duplicated a second time in the same file | Add it here too |
| 5 | [`run_inference_only.py:386`](pytorch_benchmark/run_inference_only.py#L386) | `ALL_MODELS = ["resnet50", "mobilenet_v3", "efficientnet_b0", "distilbert", "tabular_deep"]` — third independent copy of the same list | Add it here too |
| 6 | [`cluster_benchmark.py:68`](pytorch_benchmark/cluster_benchmark.py#L68) and [`cluster_benchmark_low_rpc.py:63`](pytorch_benchmark/cluster_benchmark_low_rpc.py#L63) | `ALL_MODELS = os.environ.get("BENCHMARK_MODELS", "resnet50,mobilenet_v3,...").split(",")` — a **fourth and fifth** independent copy, this time as a comma-joined string default | Add it to both env-var defaults |

That's the same five-item model name list maintained independently in five
places (`AVAILABLE_MODELS` is the sixth, but it's metadata-only and not
consulted by any of the other five to build their lists). Miss one and the
model is either unselectable from that entrypoint's CLI, or silently
missing from that script's benchmark run — no error, just absence.

**One more inconsistency on top of this:** `cluster_benchmark_low_rpc.py`
doesn't even route vision models through the shared loader —
[lines 223-225](pytorch_benchmark/cluster_benchmark_low_rpc.py#L223-L225)
call `torchvision.models.resnet50(weights=...)` etc. directly, bypassing
`load_pretrained_model()` entirely. So a new vision model added to
`AVAILABLE_MODELS` + the if/elif in `pretrained_models.py` still wouldn't
work in the low-RPC cluster script without a **separate**, differently-shaped
edit there.

---

## What a real "register once" framework looks like

The fix is to make `AVAILABLE_MODELS` the actual single source of truth,
not just a metadata sidecar next to hardcoded logic. Concretely:

**1. Store the loader in the registry entry, not in an if/elif:**
```python
AVAILABLE_MODELS = {
    "resnet50": {
        "type": "vision",
        "description": "ResNet-50 (ImageNet pretrained)",
        "input_size": (3, 224, 224),
        "num_classes": 1000,
        "loader": load_resnet50,          # <-- new
    },
    ...
}

def load_pretrained_model(model_name, device=None, **kwargs):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    model = AVAILABLE_MODELS[model_name]["loader"](**kwargs)
    ...
```
Adding a model becomes: write `load_vit_b16()`, add one dict entry. No
if/elif edit needed — item 2 above disappears entirely.

**2. Derive every CLI choices list from `AVAILABLE_MODELS.keys()`:**
```python
parser.add_argument("--models", nargs="+", choices=list(AVAILABLE_MODELS.keys()), ...)
```
in `main.py`, `run_inference_only.py`. Items 3, 4, 5 above disappear —
there's nothing left to keep in sync by hand.

**3. Same for the env-var defaults** in `cluster_benchmark.py` /
`cluster_benchmark_low_rpc.py`:
```python
ALL_MODELS = os.environ.get("BENCHMARK_MODELS", ",".join(AVAILABLE_MODELS.keys())).split(",")
```
Item 6 disappears.

**4. Route `cluster_benchmark_low_rpc.py`'s vision-model construction
through `load_pretrained_model()` instead of calling `torchvision.models.*`
directly** — removes the sixth, separately-shaped touch point and the risk
of the two cluster scripts diverging further on model behavior.

After this, adding a model is: **write one loader function, add one dict
entry.** Every CLI flag, every cluster script, and every benchmark
entrypoint picks it up automatically.

---

## The separate axis: new *distributed job types*, not just new models

Registering a model is the easy half. The harder axis — covered in
[`OFFLINE_CLUSTER_SINGLE_IMAGE.md §6`](OFFLINE_CLUSTER_SINGLE_IMAGE.md#6-caveat-which-script-decides-cpu-vs-gpu-differs) —
is that *how* a job gets distributed (Spark-resource-aware vs. hardcoded
partition-id logic) is still decided per-script, not per-model. A new
model dropped into the registry above inherits whichever distribution
strategy the script you run it from already has — it doesn't get to declare
its own placement/resource needs. If future models have different resource
profiles (e.g. a model that genuinely needs 2 GPUs, or one that should
never run on a CPU worker), that's not solved by the registry fix above —
it needs the model registry to also carry resource requirements, and the
Spark GPU-discovery wiring from the previous doc to actually exist so those
requirements can be enforced by Spark rather than assumed by convention.

---

## Recommendation

Do the registry fix (§ above) before adding the next model — it's small,
low-risk, and every model added *after* it inherits automatic CLI/script
coverage instead of needing the same 5-place edit repeated. Adding models
*before* fixing this just grows the number of places that can silently
drift out of sync.
