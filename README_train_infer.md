# SDM-EXR LINO–SDM Inference and Comparison Guide

This guide covers the private LINO training workflow on the
`dev-lino-sdm-comparison` branch and the released SDM-style EXR workflow
below. It starts with environment installation and
ends with paired LINO/SDM angular-error results.

## Released-checkpoint comparison (inference only)

There is no training command in this comparison workflow. It uses released
LINO-UniPS and SDM-UniPS checkpoints and does not retrain or change either
model architecture. The separate private-training workflow is documented
below so that this heading does not imply that LINO training is unsupported.

## Private LINO training on corrected EXR data

This workflow trains the exact released `LiNo_UniPS` normal graph on the
private `.data` objects used by the corrected SDM comparison. It keeps the
source observations at 256x256, uses LINO's native 512x512 internal path, and
does not modify the released model parameter names or the original synthetic
training entry point. The primary experiment is a cold start for 100 total
epochs; no official accuracy result exists until that run and its paired
inference have actually completed.

### Install and verify the training inputs

Run these commands from the LINO checkout. Use the LINO environment, not the
SDM environment:

```bash
cd /mnt/16TData/minhnv/LINO
conda activate LINO
python -m pip install -r requirements.txt

test -d /mnt/18TData/minhnv/train
test -d /mnt/18TData/minhnv/test
test -d /mnt/18TData/minhnv/inference
test -f output/sdm_lino_comparison/external/selected_lights.json
```

The training preset expects each train/test object to contain finite
`image*.exr` observations, `local_normal.exr`, and an independently generated
`binary_mask.exr`, all at 256x256. The final-selection manifest is used only
to bind the later 16-light comparison; it must exist before training starts.
It is the canonical manifest written by the earlier seeded comparison run. If
the check fails, locate any prior manifest with:

```bash
find output -name selected_lights.json -print
```

Use only the manifest from the intended `sdm_lino_comparison/external`
protocol; do not substitute an unrelated selection.

### Cold-start training

The checked-in preset is the primary run:

```bash
python train_private.py --config configs/lino_private_train_fixed.yaml
```

It uses six seeded lights for training, the explicit train/test roots, BF16
CUDA, AdamW, StepLR, and 100 total epochs. The command prints one line after
each successfully published epoch. Do not treat a stopped or failed run as a
completed experiment; the last valid `last.ckpt` remains the resume point.

Startup uses a filename-only structural index. It checks safe object and file
names but does not decode or hash the complete EXR dataset before CUDA is
initialized. Each consumed sample then reads and validates exactly eight files:
six epoch-selected observations, `local_normal.exr`, and `binary_mask.exr`.
There is intentionally no cross-run content ledger, so changing bytes under an
unchanged filename does not invalidate a resume fingerprint; those bytes are
still validated when the sample is consumed.

LINO derives GT support with the same `sdm_corrected_v2_unit_band` rule as the
already-trained SDM corrected-v2 baseline: finite float32 normal length with
`abs(length - 1) < 0.5`. Every such GT-valid pixel must remain inside the
external mask, while external-mask-only halo is model context without target
loss. Checkpoints and exports identify this workflow as
`lino-private-exr-training-v2`. SDM does not require retraining for this
LINO-only loading and validity correction.

### Resume from a full training checkpoint

`.ckpt` is the only artifact that resumes optimizer, scheduler, epoch, best
metric, and RNG state. A resume target is the total epoch count, not the number
of additional epochs. Copy the preset and change only the startup paths and
the total target:

```bash
cp configs/lino_private_train_fixed.yaml /tmp/lino_private_train_resume.yaml
```

Set these values in `/tmp/lino_private_train_resume.yaml`:

```yaml
startup_mode: "resume"
init_checkpoint: null
resume_checkpoint: "./runs/lino_private_fixed_lazy_sdmvalid_bf16/checkpoints/last.ckpt"
epochs: 100
```

Then run:

```bash
python train_private.py --config /tmp/lino_private_train_resume.yaml
```

For example, a checkpoint completed through epoch 40 with `epochs: 100`
continues at epoch 41. Compatibility fingerprints reject changes to the
architecture, data contract, preprocessing, light schedule, objective,
optimizer, or scheduler. Increasing the total epoch target is allowed.

### Initialize from a model-only `.pth`

Initialization is a separate experiment. It loads model weights only and
resets optimizer, scheduler, epoch, best metrics, and RNG state; it is not a
resume. Copy the preset and set:

```bash
cp configs/lino_private_train_fixed.yaml /tmp/lino_private_train_init.yaml
```

Then edit `/tmp/lino_private_train_init.yaml` with:

```yaml
startup_mode: "init_checkpoint"
init_checkpoint: "./checkpoints/lino.pth"
resume_checkpoint: null
save_dir: "./runs/lino_private_init"
epochs: 100
```

Then run it with an intentionally separate `save_dir`:

```bash
python train_private.py --config /tmp/lino_private_train_init.yaml
```

The released checkpoint route permits only the four documented author-extra
keys. Any missing current key, unexpected key, or shape/dtype mismatch fails
before the live model or CUDA device is created.

### Artifacts and selection policy

The primary run writes under `runs/lino_private_fixed_lazy_sdmvalid_bf16/`:

```text
config.resolved.yaml       # resolved training settings
data_contract.json         # data, architecture, and comparison contract
metrics.csv                # one row per successfully published epoch
checkpoints/last.ckpt      # latest complete resume state
checkpoints/best_validation.ckpt
checkpoints/lino_epoch_100.ckpt
exports/lino_epoch_100.pth # raw model-only inference weights
exports/lino_epoch_100.json
```

The adjacent export JSON is mandatory for strict trained-checkpoint
inference. The raw `.pth` contains only detached CPU tensors from the released
model; it has no `net.` or `model.` wrapper prefix and cannot resume training.
Epoch 100 is the primary comparison checkpoint. `best_validation` is a
secondary diagnostic and must not silently replace epoch 100 in the final
comparison. The training command does not produce a final comparison MAE;
that number is available only after successful inference on the final data.

### Strict inference of the epoch-100 export

After the 100-epoch run succeeds, use the paired preset:

```bash
python eval.py --config configs/lino_private_infer_trained_fixed.yaml
```

This validates the adjacent JSON sidecar, its checkpoint digest, architecture
schema, preprocessing version, 256x256 source geometry, external-mask and
unsigned-normal contract, and the exact 16-light manifest before constructing
the model. It writes signed 256x256 predictions under
`output/lino_private_trained_lazy_sdmvalid/external/lino/` and prints elapsed time,
MAE, and CUDA memory. `run.json` is created only after every selected object
finishes successfully; a missing `run.json` means the inference run is partial
or failed. Do not create it manually.

### Short manual GPU smoke gate (non-comparable)

Before committing to the full run, use an isolated one-epoch smoke config. It
uses the same code path but limits the run to the first two manifest-ordered
objects and writes below a separate `save_dir/smoke` directory:

First create a fresh, two-object inference input and matching selection
manifest. The destination data root and temporary manifest must be new/empty;
the commands below refuse to overwrite either one. They copy regular object
directories from the real final inference root, reject source symlinks, and
write a manifest containing exactly two object keys. Each copied object keeps
the exact ordered list from the full selected-lights JSON, with exactly 16
unique light names. Do not replace this with a symlink or point smoke
inference at the final eight-object root.

```bash
python - <<'PY'
import json
import shutil
from pathlib import Path

source_root = Path("/mnt/18TData/minhnv/inference")
smoke_root = Path("/mnt/18TData/minhnv/inference_smoke")
full_selection_path = Path(
    "/mnt/16TData/minhnv/LINO/output/lino_private_transfer/external/"
    "selected_lights.json"
)
smoke_selection_path = Path("/tmp/lino_private_smoke_selected_lights.json")

if smoke_root.exists() or smoke_selection_path.exists():
    raise SystemExit(
        "Refusing to overwrite smoke destinations; choose new/empty paths: "
        f"{smoke_root} and {smoke_selection_path}"
    )
if not source_root.is_dir() or not full_selection_path.is_file():
    raise SystemExit("Final inference root or full selected-lights JSON is missing")

payload = json.loads(full_selection_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or len(payload) < 2:
    raise SystemExit("Full selected-lights JSON must contain at least two object keys")
if any(
    not isinstance(name, str)
    or Path(name).name != name
    or "\\" in name
    or name in {"", ".", ".."}
    for name in payload
):
    raise SystemExit("Full selected-lights JSON contains an unsafe object key")
object_names = sorted(payload)[:2]
smoke_payload = {}

for object_name in object_names:
    values = payload[object_name]
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise SystemExit(f"{object_name} must contain a list of light names")
    if (
        len(values) != 16
        or len(set(values)) != 16
        or any(
            Path(value).name != value
            or "\\" in value
            or value in {"", ".", ".."}
            for value in values
        )
    ):
        raise SystemExit(
            f"{object_name} must preserve exactly 16 unique light names"
        )
    source_object = source_root / object_name
    if not source_object.is_dir() or source_object.is_symlink():
        raise SystemExit(f"Missing or symlinked source object: {source_object}")
    if any(path.is_symlink() for path in source_object.rglob("*")):
        raise SystemExit(f"Source object contains symlinks: {source_object}")
    for value in values:
        source_image = source_object / value
        if not source_image.is_file() or source_image.is_symlink():
            raise SystemExit(f"Missing or symlinked selected image: {source_image}")
    smoke_payload[object_name] = list(values)

smoke_root.mkdir()
for object_name in object_names:
    shutil.copytree(
        source_root / object_name,
        smoke_root / object_name,
        symlinks=False,
    )
    if any(path.is_symlink() for path in (smoke_root / object_name).rglob("*")):
        raise SystemExit(f"Copied smoke object contains a symlink: {object_name}")

smoke_selection_path.write_text(
    json.dumps(smoke_payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"Copied exactly two objects to {smoke_root}")
print(f"Wrote matching 16-light manifest to {smoke_selection_path}")
PY
```

```bash
cp configs/lino_private_train_fixed.yaml /tmp/lino_private_smoke.yaml
sed -i \
  -e 's#save_dir: "./runs/lino_private_fixed_lazy_sdmvalid_bf16"#save_dir: "./runs/lino_private_smoke"#' \
  -e 's#final_selection_manifest: "./output/sdm_lino_comparison/external/selected_lights.json"#final_selection_manifest: "/tmp/lino_private_smoke_selected_lights.json"#' \
  -e 's/epochs: 100/epochs: 1/' \
  /tmp/lino_private_smoke.yaml
python train_private.py --config /tmp/lino_private_smoke.yaml --smoke
```

Verify finite metrics, `runs/lino_private_smoke/smoke/checkpoints/last.ckpt`,
and the raw export plus adjacent JSON. Then test epoch-boundary resume:

```bash
cp /tmp/lino_private_smoke.yaml /tmp/lino_private_smoke_resume.yaml
sed -i \
  -e 's/startup_mode: "cold_start"/startup_mode: "resume"/' \
  -e 's#resume_checkpoint: null#resume_checkpoint: "./runs/lino_private_smoke/smoke/checkpoints/last.ckpt"#' \
  -e 's/epochs: 1/epochs: 2/' \
  /tmp/lino_private_smoke_resume.yaml
python train_private.py --config /tmp/lino_private_smoke_resume.yaml --smoke
```

The second command must begin at epoch 2 and append, rather than repeat,
the epoch-1 metrics row. Every smoke checkpoint/export sidecar must identify
`run_kind: smoke` and `comparable: false`.

For the smoke inference check, copy the paired inference preset and override
the checkpoint, `data_root`, `selection_manifest`, and output root. The two
overrides intentionally point to the fresh two-object root and its matching
manifest created above; both objects have exactly 16 unique light names in the
same order as the full selected-lights JSON. Keep the output outside the final
comparison directory, and explicitly opt into the non-comparable smoke
contract:

```bash
cp configs/lino_private_infer_trained_fixed.yaml /tmp/lino_private_smoke_infer.yaml
sed -i \
  -e 's#checkpoint: "./runs/lino_private_fixed_lazy_sdmvalid_bf16/exports/lino_epoch_100.pth"#checkpoint: "./runs/lino_private_smoke/smoke/exports/lino_epoch_002.pth"#' \
  -e 's#data_root: "/mnt/18TData/minhnv/inference"#data_root: "/mnt/18TData/minhnv/inference_smoke"#' \
  -e 's#selection_manifest: "./output/sdm_lino_comparison/external/selected_lights.json"#selection_manifest: "/tmp/lino_private_smoke_selected_lights.json"#' \
  -e 's#output_root: "./output/lino_private_trained_lazy_sdmvalid"#output_root: "./output/lino_private_smoke"#' \
  -e 's/require_checkpoint_data_contract: true/require_checkpoint_data_contract: true\nallow_non_comparable_checkpoint: true/' \
  /tmp/lino_private_smoke_infer.yaml
python eval.py --config /tmp/lino_private_smoke_infer.yaml
```

Verify signed 256x256 `normal_pred.exr` files, the printed inference time and
VRAM fields, and the smoke `run.json`. Smoke outputs are acceptance evidence
only: comparison scoring rejects them, and they must never be reported as the
official SDM-versus-LINO result.

## 1. Clone LINO and install its environment

The pinned packages in `requirements.txt` use CUDA 12.4 builds, including
PyTorch, xFormers, and `spconv_cu124`. Use a compatible NVIDIA driver/GPU.

```bash
git clone -b dev-lino-sdm-comparison https://github.com/Min-KiD/LINO_UniPS.git
cd LINO_UniPS

conda create -n LINO python=3.10 -y
conda activate LINO
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The SDM repository needs its own working Python environment and dependencies.
Do not install the SDM dependencies into the LINO environment unless that is
already how your machine is configured.

## 2. Define the local paths used below

Run this from the LINO repository and replace every `/absolute/path/...`
value. These shell variables make the later commands easier to copy.

```bash
export LINO_REPO="$(pwd)"
export SDM_REPO="/absolute/path/to/PhotometricStereo"
export LINO_CKPT="/absolute/path/to/lino.pth"
export SDM_CKPT="/absolute/path/to/optimized_sdm.pth"
export SDM_PYTHON="/absolute/path/to/sdm/environment/bin/python"
export CONFIG="$LINO_REPO/configs/sdm_exr_infer.yaml"
```

Required local assets:

- `LINO_CKPT`: the released LINO normal-estimation checkpoint.
- `SDM_CKPT`: the SDM checkpoint to compare against LINO.
- `SDM_REPO`: the SDM repository containing `main.py` and
  `configs/baseline_optimized_infer.yaml`.
- `SDM_PYTHON`: the executable from the working SDM environment.

Confirm the paths before continuing:

```bash
test -f "$LINO_CKPT" &&
test -f "$SDM_CKPT" &&
test -f "$SDM_REPO/main.py" &&
test -f "$SDM_REPO/configs/baseline_optimized_infer.yaml" &&
test -x "$SDM_PYTHON" &&
echo "All required paths exist."
```

If the confirmation is not printed, at least one path must be corrected.

## 3. Prepare the SDM-style EXR dataset

`data_root` must contain one directory per object. With the checked-in
defaults, object directories end in `.data`, observations begin with `image`,
and all images are three-channel EXRs.

```text
/absolute/path/to/test/
├── object_a.data/
│   ├── image01.exr
│   ├── image02.exr
│   ├── ...
│   ├── image16.exr
│   ├── binary_mask.exr
│   └── local_normal.exr
└── object_b.data/
    ├── image01.exr
    ├── ...
    ├── image16.exr
    ├── binary_mask.exr
    └── local_normal.exr
```

Important requirements:

- Every object needs at least `max_image_num` matching observation EXRs.
- Observations, mask, and normal must have the same source resolution.
- `local_normal.exr` is used only for scoring and is hidden from model inputs.
  Set `normal_encoding: signed` for XYZ values already in signed coordinates, or
  `normal_encoding: unsigned` for [0,1] values decoded as `2 * encoded - 1`.
- `binary_mask.exr` is required when `mask_policy: "external"`.
- The mask is not supplied when `mask_policy: "full"`.

## 4. Edit the single comparison YAML

Edit:

```text
configs/sdm_exr_infer.yaml
```

At minimum, replace these values:

```yaml
checkpoint: "/absolute/path/to/lino.pth"
data_root: "/absolute/path/to/test"
output_root: "./output/sdm_lino_comparison"
max_image_num: 16
mask_policy: "external"
precision: "bf16"
device: "auto"
```

Shell variables such as `$LINO_CKPT` are not expanded inside YAML. Enter the
real path in the file.

Use only this one YAML:

- `mask_policy: "external"` uses `binary_mask.exr` as model input.
- `mask_policy: "full"` uses an all-ones model input and provides no mask to
  SDM.
- `light_selection: "seeded"` creates a deterministic selected-light manifest.
- `light_selection: "manifest"` reuses the exact file named by
  `selection_manifest` for a pinned final experiment.

The selected policy is automatically appended to `output_root`. For example:

```text
./output/sdm_lino_comparison/external/
./output/sdm_lino_comparison/full/
```

## 5. Run released-checkpoint LINO inference

Run this in the LINO environment:

```bash
cd "$LINO_REPO"
conda activate LINO
python eval.py --config "$CONFIG"
```

The command reports object 1, every 100 completed objects, and the final
object with elapsed time and ETA. On success it prints the LINO macro-object
MAE and total end-to-end inference time. This immediate MAE is computed only
after each prediction and never supplies ground truth to the model.

Example completion output:

```text
LINO progress: 2399/2399 | elapsed 38:05:57 | ETA 00:00:00
Inference complete: 2399 objects -> output/sdm_lino_comparison/external/lino
Mean MAE (2399 objects): 0.7123
Total inference time: 38:06:41
```

Runtime and VRAM are not compute-matched between the released pipelines: this
LINO route can process a square up to 2048x2048 in 512x512 tiles, while the
current optimized SDM preset uses 256x256. Treat timing as released-pipeline
timing, not an architecture-only speed comparison.

A successful run writes one prediction directory per object:

```text
<output_root>/<mask_policy>/lino/<object>.data/normal_pred.exr
<output_root>/<mask_policy>/lino/<object>.data/normal_pred.png  # when save_png: true
```

`normal_pred.exr` is the authoritative signed, source-resolution prediction.
When enabled, `normal_pred.png` is only a preview and is never used for
scoring.

## Private 256x256 EXR transfer gate (before training)

Run this dedicated, inference-only route before planning any training. It fixes
the private HDR transfer contract at exactly 16 seeded lights, an external
mask, unsigned source normals, 256x256 source geometry, one 512x512 model tile,
bf16 precision, and CUDA. The output root is isolated from the comparison
workflow. It uses the released LINO checkpoint and does not run training,
`compare_sdm.py`, SDM inference, or an SDM comparison.

Use the following commands from the LINO checkout:

```bash
cd /mnt/16TData/minhnv/LINO
conda activate LINO
python -m pip install -r requirements.txt

mkdir -p checkpoints
if [ ! -s checkpoints/lino.pth ]; then
  wget -c \
    "https://huggingface.co/houyuanchen/lino/resolve/main/lino.pth" \
    -O checkpoints/lino.pth
fi
test -s checkpoints/lino.pth

test -d /mnt/18TData/minhnv/inference
python eval.py --config configs/lino_private_transfer.yaml
```

The successful output includes one signed prediction per object and a
completion marker, for example:

```text
output/lino_private_transfer/external/lino/alpha.data/normal_pred.exr  # example object
output/lino_private_transfer/external/lino/run.json
```

`normal_pred.exr` stays signed and identity-coordinate. `run.json` is created
only after all objects finish and contains source, checkpoint, config, and
output hashes; selected lights; per-object mask, HDR, and geometry diagnostics;
official MAE; the constant baseline; the coordinate diagnostic; elapsed time;
and peak CUDA memory. The 256x256 source geometry and 512x512 model geometry
are deliberately different. The coordinate diagnostic is for convention
diagnosis only and cannot be quoted as the official model MAE. This command is
inference-only: it does not run `compare_sdm.py` or SDM.

Interpret the transfer results with this decision table:

| Observation | Decision |
|---|---|
| Best coordinate MAE is materially lower than identity | Resolve the coordinate convention before training. |
| Identity is close to best and near/worse than the constant baseline | Released-checkpoint transfer failed; design LINO training/fine-tuning next. |
| Identity clearly beats the baseline but misses the project target | Transfer is partial; use the checkpoint as the fine-tuning start. |
| Identity is acceptable | Postpone training and design a separately controlled SDM-versus-LINO comparison. |

No numeric pass threshold is imposed for this private HDR distribution.

### Manifest-pinned companion for corrected SDM pairing

The released-checkpoint LINO external result (`17.6228` MAE) remains valid.
For a reproducible pairing with corrected SDM inference, rerun LINO with:

```bash
python eval.py --config configs/lino_private_transfer_fixed.yaml
```

This companion does not train LINO and is not a model fix. It changes only the
output root and replaces seeded selection with the exact ordered light list
already written by the successful transfer run:

```text
output/sdm_lino_comparison/external/selected_lights.json
```

The completion record is written only after all objects finish:

```text
output/lino_private_transfer_fixed/external/lino/run.json
```

Corrected SDM inference consumes the same manifest, so both methods see the
same sixteen observations in the same order. This remains a comparison of a
private-trained SDM checkpoint against zero-shot released LINO weights; final
architecture fairness requires training LINO on the same private split and
protocol.

## Where `run.json` is created

LINO writes this file only after every object finishes successfully:

```text
<output_root>/<mask_policy>/lino/run.json
```

With the default relative output and external-mask policy, it is:

```text
output/sdm_lino_comparison/external/lino/run.json
```

`run.json` is runtime provenance, not a source file or checkpoint. It records
the LINO config, checkpoint and manifest hashes, selected inputs, output EXR
hashes, device/precision, and runtime information. A failed or interrupted run
intentionally leaves no valid `run.json`, so a partial run cannot be scored as
complete.

Check it after inference:

```bash
test -f "output/sdm_lino_comparison/external/lino/run.json" \
  && echo "LINO run complete"
```

Adjust the path if `output_root` or `mask_policy` differs.

## 6. Prepare the GT-hidden SDM request

Stay in the LINO environment. This command validates the current assets,
persists the exact selected-light manifests, creates a ground-truth-hidden SDM
input view, and creates a fresh request/output directory.

```bash
cd "$LINO_REPO"
conda activate LINO

python compare_sdm.py prepare-view \
  --config "$CONFIG" \
  --sdm-repo "$SDM_REPO" \
  --sdm-checkpoint "$SDM_CKPT" \
  --sdm-python "$SDM_PYTHON"
```

`prepare-view` does not run SDM inference. It prints three important lines:

1. The generated GT-hidden SDM input-view path.
2. The exact SDM inference command, beginning with `SDM_PYTHON`.
3. The exact LINO-environment `finalize-sdm` command containing the generated
   request JSON path.

Keep this terminal output. The request ID is unique for every preparation.

## 7. Run the exact printed SDM command

Copy the complete SDM inference command printed by Step 6 and execute it
exactly. Do not reconstruct or shorten it. It includes the shared selected
lights, mask policy, GT-hidden input view, fresh output directory, seed, and
other fairness-critical arguments.

The command already starts with the absolute `SDM_PYTHON` executable, for
example:

```text
/absolute/path/to/sdm/environment/bin/python \
  /absolute/path/to/PhotometricStereo/main.py infer ...
```

The omitted arguments represented by `...` are required. Use the actual
printed command, not this abbreviated example.

Wait for SDM inference to finish before continuing.

## 8. Finalize the SDM result

Return to the LINO environment and run the complete `finalize-sdm` command
printed by Step 6. Its form is:

```bash
cd "$LINO_REPO"
conda activate LINO

python compare_sdm.py finalize-sdm \
  --config "$CONFIG" \
  --request "/absolute/generated/path/sdm_requests/<request-id>.json"
```

Use the exact generated request path from your terminal. Do not type the
literal `<request-id>` text.

Finalization checks the request, current source data, GT-hidden view, SDM
runtime inputs, output allowlist, EXR geometry, finite values, and prediction
hashes. On success it prints the completion JSON path.

## 9. Score LINO against SDM

Set `REQUEST` to the same request JSON path used for finalization:

```bash
export REQUEST="/absolute/generated/path/sdm_requests/<request-id>.json"
```

Then run:

```bash
cd "$LINO_REPO"
conda activate LINO
python compare_sdm.py score --config "$CONFIG" --request "$REQUEST"
```

The command prints:

- LINO macro mean angular error (MAE).
- SDM macro MAE.
- LINO minus SDM macro MAE.

Both models are scored from signed source-resolution EXRs using the same valid
source-normal support.

This paired score remains the authoritative comparison even though LINO now
prints its standalone MAE after `eval.py`. It additionally validates that LINO
and SDM used the intended selected lights, mask policy, checkpoints,
manifests, requests, hashes, and finalized prediction artifacts.

## 10. Output inventory

All paths below are relative to:

```text
<policy_root> = <output_root>/<mask_policy>
```

```text
<policy_root>/
├── input_manifest.json
├── selected_lights.json
├── lino/
│   ├── run.json
│   └── <object>.data/
│       ├── normal_pred.exr
│       └── normal_pred.png            # when save_png: true
├── sdm_input/
│   └── <object>.data/                 # GT-hidden linked observations
├── sdm_requests/
│   └── <request-id>.json
├── sdm/
│   └── <request-id>/
│       ├── <object-stem>_pred.exr
│       └── ...
├── sdm_completions/
│   └── <request-id>.json
└── comparison/
    ├── per_object.csv
    └── summary.json
```

Important files:

- `selected_lights.json`: exact ordered observations shared by LINO and SDM.
- `lino/run.json`: successful LINO-run provenance.
- `sdm_requests/<request-id>.json`: immutable instructions for one SDM run.
- `sdm_completions/<request-id>.json`: finalized SDM prediction hashes.
- `comparison/per_object.csv`: per-object LINO and SDM angular metrics.
- `comparison/summary.json`: aggregate metrics and artifact provenance.

## 11. Run both mask protocols

### External-mask comparison

Set:

```yaml
mask_policy: "external"
```

Then repeat Steps 5–9. Results go beneath `<output_root>/external/`.

### Mask-free comparison

Change only:

```yaml
mask_policy: "full"
```

Then repeat Steps 5–9. Results go beneath `<output_root>/full/` and do not
overwrite the external-mask run.

### Pin the selected lights for a final repeated experiment

The first seeded run creates:

```text
<output_root>/<mask_policy>/selected_lights.json
```

For a final repeated run, edit the same YAML:

```yaml
light_selection: "manifest"
selection_manifest: "/absolute/path/to/selected_lights.json"
```

Rerun LINO inference and prepare a new SDM request. This prevents an accidental
change in selected-light order or membership.

## 12. Troubleshooting

### `run.json` is missing

LINO did not complete every object, the process was interrupted, or a rerun
invalidated the previous provenance before failing. Read the first LINO error,
fix it, and rerun Step 5. Do not create `run.json` manually.

### OpenCV reports that EXR is disabled or unsupported

The code enables `OPENCV_IO_ENABLE_OPENEXR=1` before importing OpenCV, but the
installed OpenCV build must still include EXR support. Confirm that the active
environment uses the pinned `opencv-python-headless` from `requirements.txt`
and that no conflicting OpenCV wheel shadows it.

### A checkpoint, SDM repository, or Python executable is rejected

Repeat the five `test` commands in Step 2 and use absolute paths. The SDM
Python path must be executable, not merely the environment directory.

### `prepare-view` reports a stale or mismatched manifest

Do not reuse manifests after changing the dataset, selected observations, or
configuration. Return to `light_selection: "seeded"` to generate a fresh
selection, rerun LINO, and prepare a new SDM request.

### Finalization or scoring rejects the request

Use the exact request path printed for the current run. Do not reuse an older
request with a new SDM output, changed checkpoint, changed source data, or
different mask policy.

### SDM output is missing

`prepare-view` only prints the SDM command. Confirm that you copied and ran the
entire printed command and waited for it to finish before `finalize-sdm`.

### Comparison files are absent

`comparison/per_object.csv` and `comparison/summary.json` are created only by
the Step 9 `score` command after both a valid LINO `run.json` and finalized SDM
completion exist.
